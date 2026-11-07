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
| `nezaret[monai]` | MONAI ≥ 1.4, and through it **PyTorch** | `nezaret.lint_compose()` on live pipeline objects, and the live test suite |
| `nezaret[all]` | both | |
| `nezaret[dev]` | pytest, PyYAML | running the tests |

```bash
pip install "nezaret[yaml]"     # most people want this one
```

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
40 KB). On any machine with access to PyPI:

```bash
pip download nezaret --no-deps -d wheels/
pip download pyyaml --no-deps -d wheels/      # optional, for YAML bundles
```

Copy `wheels/` across and:

```bash
pip install --no-index --find-links wheels/ nezaret pyyaml
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
