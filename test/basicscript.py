""" 
This script demonstrates the first steps of riser profile work:
    1. Provide names of all profiles and the names of the .csv files that they are stored in.
     read_profile_files extracts each profile  as a separate pandas dataframe, stored in a list.
     The second output consists of a list of profile names. This list may be different from the input name list
     if no data was found for any profile name.
    2. split_and_center_profiles: Remove unwanted endpoints from profiles, or split a profile into multiple parts.
     In a second step, each profile is centered such that the approximate riser midpoint is located at (0, 0). This
     is a prerequisite for applying the scarp diffusion equation (Hanks and Andrews 1989).

Make sure that your current working directory is set to the parent directory of the directory containing your data files.
Use os.chdir("path/to/your/dir") if necessary.
"""

import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../src/'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
os.chdir("C:/Users/Lennart/lennartGit/riserfitTesting")

import riserfit as rfc
import matplotlib.pyplot as plt
import numpy as np

# name of subdirectory containing your .csv data files
filepath = "Data//Patagonia_2023_Riser_Data"

# names of .csv files in the data directory
infiles = ["060323-SantaCruz.csv", "070323-SantaCruz.csv",
           "080323-SantaCruz.csv", "260223-TresLagos.csv",
           "270223-TresLagos.csv", "280223-TresLagos.csv"]

# profile names contained in the .csv files. Each data point
# must be associated with a profile name.
profilenames = [
    "280223T5T3f",
    "280223T3T5b",
    "280223T3T5c",
    "280223T2T3a",
    "280223T2T3b",
    "280223T2T3c",
    "280223TT4T6",
    "280223T4T6b",
    "260223T2T5a",
    "260223T2T5c",
    "270223T5T6c",
    "270223T5T6d",
    "270223T6T7a",
    "270223T6T7b",
    "270223T6T7c",
    "270223T6T7d",
    "260223T0T2a",
    "260223T0T2b",
    "260223T0T2c",
    "270223T6T7e",
    "270223T6T7f"
]
# header of the column in .csv files containing the profile name
identifier = "Name"  

# preprocessing steps:
# read profiles and store in list
profiles, names = rfc.read_profile_files(filepath,
    infiles,
    profilenames,
    identifier
)
# run splitting and centering routine
cprofiles, cnames = rfc.split_and_center_profiles(profiles,
    names,
    summarypdf=False
)  
# initialize riser class
risers = rfc.initialize_riser_class(cprofiles, cnames)

# run optimization routine
risers.compute_best_linear_diffusion_fit(summarypdf=False)

# add CN ages

risers.add_cn_ages(filepath="Data/Patagonia_2023_Riser_Data",
    infile="MatchingProfilesAndCN.csv",
    reliability="Caution",
    reliability_dict={"y": "gray", "n": "blue"},
    kt_cn_plot=True,
    sep=";",
    decimal=","
)

# compute kt misfit

risers.calculate_kt_uncertainty(
    frac=0.5,
    maxfev=10000, # this doesn't help, but just keep it.
    reliability_dict={"y": "gray", "n": "blue"},
    set_lb_zero=True, reliability_title="Caution",
    annotate_points=False,
    ascending=True
)

# compute gradient (just out of curiosity)

risers.compute_gradients()

# compute the nonlinear diffusion stuff

risers.apply_d_z_offsets()
risers.compute_best_nonlinear_diffusion_fit()

# showcase animation

risers.animate_nonlin_profile_diffusion(name=names[5])
