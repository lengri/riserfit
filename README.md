# RiserfitTesting

__riserfitclone__ is a Python package for analyzing and age-dating fluvial terrace risers. It provides functions and classes to pre-process data, calculate diffusion ages, and compare those ages against cosmogenic nuclide (or other) ages.

## Installation

To install this package in its own virtual environment on Windows, type
```
python -m venv riserfitEnv
cd riserfitEnv/Scripts
activate

pip install git+https://git-int.gfz-potsdam.de/lgrimm/riserfitTesting.git
```
Installation on Linux-based distributions should be similar:
```
python -m venv riserfitEnv
source riserfitEnv/bin/activate

pip install git+https://git-int.gfz-potsdam.de/lgrimm/riserfitTesting.git
```
## Verifying the installation
You can test the installation by running Python from the console and importing
the riserfitclone package:
```
python
import riserfit.profiles
```
## Directory structure for a project

Depending on what you aim to use this package for, different project structures are recommended. In all cases, riserfit uses the current working directory to search for files or sub-directories.

### Analyzing GPS profiles
If you want to analyze individual profiles using riserfit.profiles, you need at least a directory containing .csv files. Also keep in mind that some functions from this package create new sub-directories and files in your project directory.
```
project_directory
|   your_script.py
|
+---profile_directory
|       profile_file1.csv
|       profile_file2.csv
|       profile_file3.csv
|       
+---cn_directory
|       cn_data.csv

```
### Working with DEMs

WIP

## Authors and acknowledgment
WIP

## License
WIP

## Project status
WIP