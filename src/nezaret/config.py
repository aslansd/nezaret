"""Reading MONAI Bundle configs without running them.

MONAI's own ``ConfigParser`` evaluates ``$`` expressions, which means
importing torch, the network code and anything the bundle's ``scripts/``
folder pulls in. nezaret never does that. It resolves only the parts of the
syntax that can be resolved by reading:

* ``@a#b#0``   a reference to another node
* ``%a#b``     a macro (a copy of another node), also ``%other.json#a#b``
* ``$@a + @b`` concatenation of referenced lists, the one expression bundles
  use to build transform pipelines

Every other ``$`` expression is left as an :class:`Unresolved` value, and the
rules treat it as unknown rather than guessing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ID_SEP = "#"
_REF = re.compile(r"@([A-Za-z0-9_#\-.]+)")
_CONCAT = re.compile(r"^\$\s*@[A-Za-z0-9_#\-.]+(\s*\+\s*@[A-Za-z0-9_#\-.]+)*\s*$")
_MAX_DEPTH = 25


class Unresolved(str):
    """An expression nezaret will not evaluate."""


class ConfigError(Exception):
    pass


def load_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised via CLI note
            raise ConfigError(
                f"{path.name}: reading YAML needs PyYAML (pip install nezaret[yaml])"
            ) from exc
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.name}: not valid JSON ({exc})") from exc


@dataclass
class ConfigFile:
    """One config file plus the files a reference may fall back to."""

    name: str
    data: Any
    path: Path | None = None
    fallbacks: list["ConfigFile"] = field(default_factory=list)
    siblings: dict[str, "ConfigFile"] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def text(self) -> str:
        return json.dumps(self.data, default=str)

    # -- lookup -----------------------------------------------------------
    def _get_local(self, ref: str):
        """Walk ``a#b#0`` through this file. Returns (found, value)."""
        node = self.data
        if isinstance(node, dict) and ref in node:  # flat override keys
            return True, node[ref]
        parts = ref.split(ID_SEP)
        for i, part in enumerate(parts):
            if isinstance(node, dict):
                rest = ID_SEP.join(parts[i:])
                if rest in node and i > 0:
                    return True, node[rest]
                if part not in node:
                    return False, None
                node = node[part]
            elif isinstance(node, list):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError):
                    return False, None
            else:
                return False, None
        return True, node

    def lookup(self, ref: str):
        """Find ``ref`` here, then in the files this one overrides."""
        found, value = self._get_local(ref)
        if found:
            return True, value, self
        for fb in self.fallbacks:
            found, value = fb._get_local(ref)
            if found:
                return True, value, fb
        return False, None, None

    def sibling(self, filename: str) -> "ConfigFile | None":
        name = Path(filename).name
        return self.siblings.get(name)

    # -- resolution -------------------------------------------------------
    def resolve(self, value, path: str = "", depth: int = 0):
        """Resolve one node shallowly.

        Returns ``(value, location, owner)`` where ``location`` is the
        ``#``-path of the node the value actually came from, so a finding on a
        transform reached through ``@train#deterministic_transforms`` points
        at that list, not at the reference.
        """
        owner = self
        if depth > _MAX_DEPTH:
            return Unresolved(str(value)), path, owner
        if not isinstance(value, str):
            return value, path, owner
        text = value.strip()
        if text.startswith("@") and _REF.fullmatch(text):
            ref = text[1:]
            found, target, where = self.lookup(ref)
            if not found:
                return Unresolved(value), path, owner
            return where.resolve(target, ref, depth + 1)
        if text.startswith("%"):
            body = text[1:]
            if ID_SEP in body and body.split(ID_SEP, 1)[0].endswith((".json", ".yaml", ".yml")):
                fname, ref = body.split(ID_SEP, 1)
                other = self.sibling(fname)
                if other is None:
                    return Unresolved(value), path, owner
                found, target, where = other.lookup(ref)
            else:
                ref = body
                found, target, where = self.lookup(ref)
            if not found:
                return Unresolved(value), path, owner
            return where.resolve(target, ref, depth + 1)
        if text.startswith("$"):
            return Unresolved(value), path, owner
        return value, path, owner

    def resolve_deep(self, value, depth: int = 0):
        """Resolve a value and everything inside it (for parameter values)."""
        value, _, owner = self.resolve(value, depth=depth)
        if isinstance(value, list):
            return [owner.resolve_deep(v, depth + 1) for v in value]
        if isinstance(value, dict):
            return {k: owner.resolve_deep(v, depth + 1) for k, v in value.items()}
        return value

    def resolve_list(self, value, path: str):
        """Resolve a ``transforms`` value to ``[(item, location, owner), ...]``.

        Handles a literal list, a reference or macro to a list, and
        ``$@a + @b`` concatenation. Returns None if it cannot be resolved.
        """
        if isinstance(value, str) and _CONCAT.match(value.strip()):
            out = []
            for ref in _REF.findall(value):
                part = self.resolve_list("@" + ref, ref)
                if part is None:
                    return None
                out.extend(part)
            return out
        resolved, loc, owner = self.resolve(value, path)
        if isinstance(resolved, str) and _CONCAT.match(resolved.strip()):
            return owner.resolve_list(resolved, loc)
        if not isinstance(resolved, list):
            return None
        return [(item, f"{loc}{ID_SEP}{i}" if loc else str(i), owner)
                for i, item in enumerate(resolved)]


# Files a bundle override config is normally merged onto, in order.
_BASES = ("train", "inference")


def load_bundle(paths: list[Path]) -> tuple[list[ConfigFile], list[str]]:
    """Load the config files of one bundle and link their fallbacks."""
    files: list[ConfigFile] = []
    notes: list[str] = []
    for p in sorted(paths):
        try:
            data = load_file(p)
        except ConfigError as exc:
            notes.append(str(exc))
            continue
        if not isinstance(data, dict):
            continue
        files.append(ConfigFile(name=p.name, data=data, path=p))
    by_name = {f.name: f for f in files}
    by_stem = {f.stem: f for f in files}
    for f in files:
        f.siblings = by_name
        f.fallbacks = [by_stem[b] for b in _BASES if b in by_stem and by_stem[b] is not f]
    return files, notes


def from_dict(data: dict, name: str = "config.json") -> ConfigFile:
    cf = ConfigFile(name=name, data=data)
    cf.siblings = {name: cf}
    return cf
