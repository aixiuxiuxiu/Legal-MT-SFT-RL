# Fine-tuning Vision Language Models (VLMs)

## Dependencies

Dependencies are managed by [uv][uv], a package/project manager for Python that is meant to bring a Cargo-style workflow
from Rust to Python. This includes managing the dependencies with a lockfile to ensure that all builds are reproducible.
While all of this was possible with virtual environments and specific tools, they always needed to be managed manually.
`uv` will automatically create a virtual environment for the project (in `.venv/`) and can then be used to run any
Python script in this environment without needing to manually activate the environment (but you can still do this if
you'd like to have everything available for your whole system).

You can install `uv` with a standalone installer (recommended), as described in the [Docs - Installing uv][uv-install],
or with pip.

```sh
pip install --upgrade uv
```

Once installed, you can manage the dependencies through `uv` as follows:

```sh
# Installs all the dependencies and automatically creates the venv
uv sync
```

In order to run a binary from an installed package in the venv without activating it, you can use `uv run`. The same
applies to running Python from the venv with `uv run python` (just prefixing it with `uv run`).

```sh
# Run the train script from the venv
uv run python train.py --help

# Run the ruff binary (installed as a dev dependency) to lint files
uv run ruff check
```

## Faster Model Downloads

The *experimental* library [`hf-transfer`][hf-transfer] allows much greater download speeds for models (beyond 500MB/s) from the
HuggingFace Hub, but may be less stable. It is included in the dependencies, but you need to enable it by setting the
environment variable `HF_HUB_ENABLE_HF_TRANSFER`

For example, if you run any script that downloads the model, you could just prepend it with the environment variable
just for that command:

```sh
HF_HUB_ENABLE_HF_TRANSFER=true uv run [...]
```

[hf-transfer]: https://github.com/huggingface/hf_transfer
[uv]: https://github.com/astral-sh/uv
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/
