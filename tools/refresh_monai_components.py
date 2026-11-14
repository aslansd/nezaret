"""Regenerate src/nezaret/data/monai_components.json from the installed MONAI.

Run this when a new MONAI release comes out, in an environment with that
release installed:

    python tools/refresh_monai_components.py

Then run the live tests (pytest tests/test_live_monai.py -v). They check the
defaults in knowledge.py against the same install, and those have to be
updated by hand: this script only refreshes the list of names.
"""

import json
from pathlib import Path

import monai
from monai.bundle.config_item import ComponentLocator

from nezaret.knowledge import ENV_DEPENDENT_COMPONENTS

OUT = Path(__file__).resolve().parent.parent / "src" / "nezaret" / "data" / "monai_components.json"


def main():
    loc = ComponentLocator()
    names = sorted(set(loc._find_classes_or_functions(loc._find_module_names()))
                   - ENV_DEPENDENT_COMPONENTS)
    old = json.loads(OUT.read_text()) if OUT.exists() else {"names": [], "monai_version": "none"}
    added = sorted(set(names) - set(old["names"]))
    removed = sorted(set(old["names"]) - set(names))
    OUT.write_text(json.dumps({"monai_version": monai.__version__.split("+")[0],
                               "source": "monai.bundle.config_item.ComponentLocator",
                               "names": names}, indent=0))
    print(f"MONAI {old['monai_version']} -> {monai.__version__}: {len(names)} names, "
          f"{len(added)} added, {len(removed)} removed")
    if removed:
        print("removed (consider adding to knowledge.REMOVED with a replacement):")
        for n in removed:
            print("  ", n)
    print("Now run tests/test_live_monai.py and update any defaults in knowledge.py it reports.")


if __name__ == "__main__":
    main()
