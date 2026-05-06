# Installation

## Install from PyPI

After the package is released on PyPI:

```bash
pip install isovae
```

## Install from GitHub

```bash
pip install git+https://github.com/ultraypy/IsoVAE.git
```

## Local development installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ultraypy/IsoVAE.git
cd IsoVAE
pip install -e .
```

## Documentation dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

Then run:

```bash
mkdocs serve
```

Open the local URL shown by MkDocs, usually:

```text
http://127.0.0.1:8000
```

## Recommended Python version

IsoVAE requires Python 3.10 or newer. Python 3.10 and 3.11 are recommended.
