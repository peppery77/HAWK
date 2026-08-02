# Contributing to HAWK

Thanks for helping improve HAWK. Bug reports, documentation fixes, benchmark adapters, and model integrations are welcome.

## Development setup

```bash
conda create -n hawk-dev python=3.10 -y
conda activate hawk-dev
pip install -r requirements.txt
make setup PYTHON_BIN="$(which python)"
make test PYTHON_BIN="$(which python)"
```

Before opening a pull request, run:

```bash
scripts/run.sh -m pytest
scripts/run.sh -m ruff check src scripts tests
```

Please keep changes focused, add tests for behavioral changes, and avoid committing checkpoints, datasets, generated evaluation outputs, credentials, or machine-specific paths.

## Reporting issues

Include the HAWK commit, Python/PyTorch/CUDA/Transformers versions, GPU model, exact command, full traceback, and whether the issue reproduces with `KEEP_RATIO=1.0`.
