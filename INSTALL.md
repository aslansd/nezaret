# Installing nezaret

## Just the tool

```bash
pip install nezaret
```

That is everything for JSON bundle configs. nezaret has **zero runtime
dependencies** and works on Python 3.10 or newer. It needs no account, no
server, no GPU, and makes no network call.

Verify:

```bash
python -c "import nezaret; print(nezaret.__version__)"
nezaret doctor
```

---

## The optional extras

| Extra | Pulls | Needed for |
|---|---|---|
| `nezaret[yaml]` | PyYAML | bundles written in YAML (`train.yaml`). Without it those files are skipped with a note, never silently |
| `nezaret[data]` | numpy, nibabel, pydicom | `nezaret data` beyond the path checks: reading NIfTI and DICOM headers and voxels. Without it, missing files and shared entries are still found and every unread file is listed in a note |
| `nezaret[result]` | numpy, nibabel, scipy | `nezaret result`: reading predictions and labels, and surface distances (scipy). Without scipy it scores Dice only and says so in a note |
| `nezaret[monai]` | MONAI ≥ 1.4, and through it **PyTorch** | `nezaret.lint_compose()` on live pipeline objects, and the live test suite |
| `nezaret[all]` | all of the above | |
| `nezaret[dev]` | pytest, PyYAML, numpy, nibabel, pydicom, scipy | running the tests (MONAI and MedPy separately, see TESTING.md) |

```bash
pip install "nezaret[yaml,data,result]"     # everything except MONAI
```

None of these pulls torch. nibabel and pydicom are pure Python on top of
numpy; scipy is the largest, at a few tens of MB.

Compressed DICOM (JPEG, JPEG 2000, JPEG-LS) also needs a decoder for the
voxel checks, e.g. `pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`.
Without one, headers are still read (geometry, patient, scanner) and the
voxel checks for those files are listed in a note.

**Linting a bundle never needs MONAI.** Bundle configs are read as data, so a
laptop with no CUDA and no torch checks them as well as a GPU server does. Only
`lint_compose()` needs MONAI, because it reads attributes off transform objects
that must already exist in your process.

PyTorch is a large download (several hundred MB, more with CUDA). If you only
want MONAI for `lint_compose()` on a CPU machine, the CPU wheel is much smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "nezaret[monai]"
```

### Which MONAI

nezaret's rules were written against **MONAI 1.6.1**, and the list of names
MONAI can resolve (for `NZ104`) was read from that release. Other MONAI
versions should work with `lint_compose()` as long as the transform classes
keep their attribute names, but only 1.6.1 has been tested. `nezaret doctor`
says when your MONAI differs from 1.6.1; the live tests then say whether
anything nezaret assumes has changed:

```bash
pytest tests/test_live_monai.py -v
```

---

## Offline or behind a slow connection

Because the core has no dependencies, the wheel is the whole install (about
75 KB). On any machine with access to PyPI:

```bash
pip download nezaret --no-deps -d wheels/
pip download pyyaml --no-deps -d wheels/      # optional, for YAML bundles
pip download numpy nibabel pydicom scipy -d wheels/  # optional, for data and result (pick your platform's wheels)
```

Copy `wheels/` across and:

```bash
pip install --no-index --find-links wheels/ nezaret pyyaml nibabel pydicom scipy
```

---

## From source

```bash
git clone https://github.com/aslansd/nezaret
cd nezaret
pip install -e ".[dev]"
pytest -q
```

See [TESTING.md](TESTING.md) for what a green run looks like in each
environment.

### When pip cannot reach PyPI

`pip install -e .` first downloads `setuptools>=77` into an isolated build
environment, so it fails with `No matching distribution found for
setuptools>=77` on a connection that times out, even though nezaret itself
needs nothing. Build with what is already installed instead (a conda `base`
usually has setuptools, pytest, PyYAML, numpy, and often nibabel):

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -c "import setuptools; print(setuptools.__version__)"   # must be 77 or newer
pip install -e . --no-build-isolation --no-deps
pytest -q
```

Without an install at all, `PYTHONPATH=src python -m pytest -q` runs every
test except the one that compares installed metadata with `__version__`.
