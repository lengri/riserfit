import sys, os
import scipy as sp
testdir = os.path.dirname(__file__)
srcdir = '../src/'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
os.chdir("C:/Users/Lennart/lennartGit/riserfitTesting")
import riserfit as rf
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from scipy.interpolate import interpn


rasterpath="\\Data\\DEM_Patagonia\\NOTFORRELEASE_EPSG32719.tif"
pointpath="\\Data\\DEM_Patagonia\\profile_points.csv"

p, n, x, y, z, id = rf.construct_z_profiles(
    rasterpath,
    pointpath,
    "name",
    "Easting",
    "Northing"
)
risers = rf.initialize_riser_class(
    p, n, x, y, z, id
)

risers.compute_best_linear_diffusion_fit(
    d_off_range=(-50, 50),
    z_off_range=(-20, 20),
    summarypdf=True
)
sys.exit()
centerpoint = (4507654, 317975)  # 317975 4507654
direction = rf.calculate_aspect(
    rasterpath, centerpoint
)[0] # unpack from list
n_points = 100
spacing = np.sqrt(2)
prof_points = rf.extrapolate_line_from_point(
    centerpoint,
    direction,
    n_points,
    spacing
)
rf.calculate_aspect(
    rasterpath,
    centerpoint
)
# get x, y coordinates for plotting:
x = [t[1] for t in prof_points]
y = [t[0] for t in prof_points]

dem = rio.open(os.getcwd()+rasterpath)
extent=dem.bounds
print(extent)
extent=[extent[0], extent[2], extent[1], extent[3]]
plt.imshow(dem.read(1), extent=extent)
plt.scatter(x, y, c="black", s=1)
plt.show()

