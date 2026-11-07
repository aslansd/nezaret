# Publishing nezaret

## The release, start to finish

```bash
pip install build twine
```

**1. Bump the version in one place.** `pyproject.toml` reads it from the source
(`dynamic = ["version"]`), so there is no second copy to forget:

```bash
# src/nezaret/__version__.py
__version__ = "0.1.1"
```

`tests/test_cli.py::test_installed_version_matches_source` fails if the
installed metadata and `nezaret.__version__` ever disagree. daftar 1.1.0 and
1.1.1 both reported themselves as 1.0.0 for exactly this reason.

**2. Write the changelog entry first, not last.** If you cannot describe the
change in a paragraph that says why it matters, the change is probably not
finished.

**3. Verify.**

```bash
pytest -q                                            # 79 passed, 3 skipped with MONAI
NEZARET_MODEL_ZOO=/tmp/model-zoo/models pytest -q    # 82 passed
nezaret lint examples/faulty_bundle                  # 5 errors, 3 warnings, 1 info
nezaret doctor
```

**4. Build and check.**

```bash
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
```

**5. Test the artifact, not the source tree.** This catches packaging mistakes
the test suite cannot see, such as `data/monai_components.json` missing from
the wheel, which would make every bare `_target_` an NZ104 error:

```bash
python -m venv /tmp/check
/tmp/check/bin/pip install dist/nezaret-*.whl pytest
/tmp/check/bin/nezaret doctor            # must say "2446 component names"
mkdir /tmp/sd && tar xzf dist/nezaret-*.tar.gz -C /tmp/sd
cd /tmp/sd/nezaret-* && /tmp/check/bin/python -m pytest -q    # 66 passed, 6 skipped
```

**6. Upload.**

```bash
python -m twine upload dist/*
git tag v0.1.1 && git push --tags
```

**7. Confirm.** pip caches the index page and PyPI's CDN takes a minute or two,
so an immediate `pip install -U nezaret` will often show the old version:

```bash
pip install --no-cache-dir -U nezaret
```

---

## When MONAI releases

nezaret's knowledge of MONAI has two parts, and they go stale differently.

**The component list** (`src/nezaret/data/monai_components.json`) is generated:

```bash
pip install -U monai
python tools/refresh_monai_components.py
```

It prints which names were added and removed. A removed transform should go
into `knowledge.REMOVED` with its replacement, so `NZ104` can say what to use.

**The defaults and transform tables** in `knowledge.py` are written by hand,
and `tests/test_live_monai.py` checks them:

```bash
pytest tests/test_live_monai.py -v
```

A failure there is a real change in MONAI (a default changed, a transform was
renamed), not a flaky test. Update `knowledge.py`, add a line to the
changelog saying which MONAI version changed what, and release. The README's
"checked against MONAI x.y.z" line and `nezaret doctor` follow the component
list automatically.

---

## Things that have cost time before

These come from publishing nezaret's sibling packages.

**A version number can never be reused**, not even after deleting the release.
If 0.1.1 turns out wrong, ship 0.1.2.

**`pip install --no-cache-dir`** when verifying. Without it you will see the
previous version and conclude the upload failed.

**Use an API token, not a password.** PyPI account settings → API tokens. Scope
it to the project after the first upload. To avoid re-pasting, put it in
`~/.pypirc` (mode 600):

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

**TestPyPI is a separate service** with separate accounts and tokens. It is
rarely worth the detour: `twine check` plus the clean-venv install in step 5
covers what a rehearsal would tell you.

---

## On the licence

`LICENSE` contains the full Apache-2.0 text and `pyproject.toml` declares
`license = "Apache-2.0"` as an SPDX expression. Keep the source public and the
licence file present.

---

## Before announcing a release publicly

- [ ] `pytest -q` green, with and without MONAI
- [ ] model-zoo regression test green against a fresh clone
- [ ] `nezaret lint examples/faulty_bundle` matches the README
- [ ] wheel installed into a clean venv, `nezaret doctor` reports 2446 names
- [ ] README renders correctly on the live PyPI page
- [ ] **nezaret run on your own MONAI work, and every finding read**
- [ ] `git tag && git push --tags`

The second-to-last item is the one that matters. A linter earns trust by what
it does *not* say; a false positive on your own pipeline, found before
announcing, is worth more than any test.
