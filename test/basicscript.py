import numpy as np
import pickle
import os, sys 
import matplotlib.pyplot as plt
testdir = os.path.dirname(__file__)
srcdir = '../'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import riserfit as rf
os.chdir("C:/Users/Lennart/lennartGit/personal/riserfit")

pg = rf.RiserPlayground()
pg.create_profiles_from_parameters(
    d=np.linspace(-100, 100, 200),
    kt=1,
    a=5,
    b=0,
    theta=0.5
)

pg.downsample_upsample_profiles(
    resample_dx=1,
    std_z=0.02
)