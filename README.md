# riserfit

__riserfit__ is a Python package for analyzing and dating fluvial terrace risers. It provides functions and classes to pre-process data from GNSS recordings and DEMs, calculate diffusion ages, and compare those ages against cosmogenic nuclide (or other) ages.

## Installation

To install this package in its own virtual environment on Windows, type
```
python -m venv riserfit
cd riserfit/Scripts
activate

pip install git+https://git-int.gfz-potsdam.de/lgrimm/riserfitTesting.git
```
Installation on Linux-based distributions should be similar:
```
python -m venv riserfit
source riserfit/bin/activate

pip install git+https://git-int.gfz-potsdam.de/lgrimm/riserfitTesting.git
```
## Verifying the installation
You can test the installation by running Python from the console and importing
the riserfit package:
```
python
import riserfit
```

