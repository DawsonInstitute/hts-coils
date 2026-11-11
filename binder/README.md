# MyBinder Configuration

This directory configures the one-click deployment of the HTS coil notebooks on mybinder.org.

## Files

- `runtime.txt` - Python 3.11
- `requirements.txt` - exactly pinned dependencies (the versions the notebooks are tested against in CI)

Because a `binder/` directory exists, repo2docker reads **only** this directory for configuration. The root
`Dockerfile`, `Dockerfile.base` and `Dockerfile.mybinder` are therefore *not* used by mybinder.org; they are for
local/Docker use.

repo2docker also runs `git submodule update --init --recursive`, so every gitlink in the tree must have a matching
entry in `.gitmodules` (currently `src/warp/optimizer`).

## Maintenance

- Notebooks locate the package via `../src` (`sys.path`), so no `postBuild` or package install is required.
- CI (`.github/workflows/notebooks.yml`) executes every notebook with these pins from a recursive clone.
- To re-pin: create a Python 3.11 venv, install the new versions, run the notebooks, then update `requirements.txt`.

## Resource constraints

MyBinder limits: ~1-2 GB RAM, 10-minute inactivity timeout, 6-hour maximum runtime, no persistent storage.
