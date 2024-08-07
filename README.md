# riserfit

__riserfit__ is a Python package for analyzing and dating fluvial terrace risers. It provides functions and classes to pre-process data from GNSS recordings and DEMs, calculate diffusion ages, and compare those ages against cosmogenic nuclide (or other) ages.

Detailed code documentation and some basic tutorials are available on [Read the Docs](https://riserfit.readthedocs.io/en/latest/) (WIP).

## Installation

To install this package in its own virtual environment on Windows, type
```
python -m venv riserfit
cd riserfit/Scripts
activate

pip install git+https://github.com/lengri/riserfit.git
```
Installation on Linux-based distributions and on macOS is similar:
```
python -m venv riserfit
source riserfit/bin/activate

pip install git+https://github.com/lengri/riserfit.git
```
## Verifying the installation
You can test the installation by running Python from the console and importing
the riserfit package:
```
python
import riserfit
```

