from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Union, Tuple
from typing_extensions import Self # pre python 3.11

import os, sys

from riserfit.profiles import *
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import itertools as it

# class for handling list/array reshaping, depending on swath_number

class Constructor_Extractor:
	"""
 	Internal class for handeling profile building and extraction from a
	DEM.
	This function should never be called by a package user.
  	"""
	def __init__(
		self,
		centerpoints: list, 
		directions: list,
		start_end_points: bool,
		n_points_array: np.ndarray,
		spacing_array: np.ndarray,
		swath_number: float,
		swath_spacing: float
	) -> None:
		"""
		Initialize a ``Constructor_Extractor`` instance.
  
		Parameters:
		-----------
			centerpoints: list
				List tuples of centerpoints :math:`(x, y)`.
			directions: list
				List of tuples of profile orientations :math:`(x, y)`.
			start_end_points: bool
				If true, the centerpoints list is interpreted as a
				list of tuples where tuple i indicates the start point, 
				and tuple :math:`i+1` indicates the end point of a profile.
			n_points_array: np.ndarray
				Number of points in each profile.
			spacing_array: np.ndarray
				The spacing between points in each profile. Note: this
				parameter is ignored if ``start_end_points == True``.
			swath_number: float
				The number of parallel profile lines for each profile.
			swath_spacing: float
				The distances between swath lines.

		Returns:
		--------
			None
		"""
		self.centerpoints = centerpoints
		self.directions = directions
		self.start_end_points = start_end_points
		self.n_points_array = n_points_array
		self.spacing_array = spacing_array
		self.swath_number = swath_number
		self.swath_spacing = swath_spacing

	def build_center_profiles(
		self
	) -> Self:
		"""
		Build profiles based on the provided data.
  
		Parameters:
		-----------
			None

		Returns:
		--------
			self: Self
				The ``Constructor_Extractor`` instance.
  		"""
		p_lines = []
		# center lines for both start_end_point true or false,
		# the correct spacing should be chosen in the dem.py script.
		for i, point in enumerate(self.centerpoints):

			# build a profile line (centerline, if swath_number > 0)
			p_line = extrapolate_line_from_point(
				centerpoint=np.array(point),
				direction=np.array(self.directions[i]),
				n_points=self.n_points_array[i],
				spacing=self.spacing_array[i]
			)

			# nested list, each sub-list contains (y, x) tuples
			p_lines.append(p_line) 

		self.centerlines = p_lines

		return self

	def build_swath_profiles(
		self
	) -> Self:

		"""
  		Build swath profile point tuples based on the input data.

		Parameters:
		-----------
			None

		Returns:
		--------
			self: Self
				The ``Constructor_Extractor`` instance.
    	"""

		# assuming that swath_number > 0...
		big_profile_list = []

		for i, centerline in enumerate(self.centerlines):

			# contains all points for a swath profile
			single_profile_list = []
			single_profile_list.append(centerline) 

			# build a list for each swath line (offset from centerline)
			# rotate direction by 90 degrees
			azimuth90 = (self.directions[i][1], -self.directions[i][0])

			for j in range(1, self.swath_number+1):

				# one subswath offset in +j*spacing*azimuth90
				subswath = [(c[0]+self.swath_spacing*j*azimuth90[0],
					c[1]+self.swath_spacing*j*azimuth90[1]) for c in centerline]
				single_profile_list.append(subswath)

				# one subswath offset in -j*spacing*azimuth90
				subswath = [(c[0]-self.swath_spacing*j*azimuth90[0],
					c[1]-self.swath_spacing*j*azimuth90[1]) for c in centerline]
				single_profile_list.append(subswath)
			
			# list of swath lines for one profile complete, append to
			# big list.
			big_profile_list.append(single_profile_list)

		self.swath_list = big_profile_list

		return self


	def build_decomposed_point_list(
		self
	) -> Self:
		"""
		Builds and unravelles a nested list containing profile lines, swaths,
		and points. The decomposed list is used for efficient 
		elevation extraction from the DEM.
  
		Parameters:
		-----------
			None

		Returns:
		--------
			self: Self
				The ``Constructor_Extractor`` instance.
  		"""
		if self.swath_number == 0:
			# simple decomposition in one step...
			# merge all p_line together, keep track of each length
			point_number = [0] + [len(prof) for prof in self.centerlines]
			point_number = list(np.cumsum(np.array(point_number)))
			points = list(it.chain.from_iterable(self.centerlines))

			self.single_line_point_number = point_number
			self.decomposed_single_line_points = points

		else:
			# the swath_list is constructed like this:
			# swath list = [s1, s2, ..., sn]
			# s1 = [l1, l2, ..., lm]
			# l1 = [(y1, x1), (y2, x2), ..., (yk,xk)]

			# decompose list
			points_temp = list(it.chain.from_iterable(self.swath_list))
			points = list(it.chain.from_iterable(points_temp))
			self.decomposed_swath_line_points = points

			# keep track of the origin of points in decomposed list
			ordered_line_len_counter = []
			n_swaths = self.swath_number*2 + 1

			for n_points in self.n_points_array:
				line_len = n_points * 2 + 1
				# stores line lines for each profile
				ordered_line_len_counter.append(line_len)

			self.ordered_line_len_counter = ordered_line_len_counter

		return self

	
	def reconstruct_elevation_list(
		self,
		z: np.ndarray
	) -> Self:

		"""
		Re-form a nested list, grouped according to profiles, swaths, points.
		This function relates to ``Constructor_Extractor.build_decomposed_point_list()``.
  
		Parameters:
		-----------
			z: np.ndarray
				Elevations extracted from the DEM.

		Returns:
		--------
			self: Self
				The ``Constructor_Extractor`` instance.
  		"""
		profile_z_list = []
		z = np.array(z)

		if self.swath_number == 0:
			slpn = self.single_line_point_number

			for i, center in enumerate(self.centerpoints):

				z_profile = list(z[slpn[i]:slpn[i+1]])
				profile_z_list.append(z_profile)
		else:

			swath_line_number = self.swath_number*2 + 1

			profile_points_indices = [0]

			# get indices of start and end for all points belonging
			# to one profile
			for i, center in enumerate(self.centerpoints):

				# get total number of points in one profile:
				n_line_points = self.ordered_line_len_counter[i]
				n_profile_points = n_line_points * swath_line_number
				profile_points_indices.append(n_profile_points)
			profile_points_indices = \
				list(np.cumsum(np.array(profile_points_indices)))
			#print(profile_points_indices)
			# extract elevation values and reshape to array with
			# shape (n_line_points, swath_line_number)
			# calculate row-wise averages, append to profile_z_list

			for i, center in enumerate(self.centerpoints):

				z_1d = z[profile_points_indices[i]:profile_points_indices[i+1]]
				shape = (self.ordered_line_len_counter[i], swath_line_number)
				z_2d = np.reshape(z_1d, shape, order="F")
				z_profile = np.mean(z_2d, axis=1)
				profile_z_list.append(z_profile)

		self.profile_z_list = profile_z_list
	
		return self


# function that reads a .csv containing a list of easting/northing values
# to be used as center points for constructing profile lines
# and outputs the profile lines

def construct_z_profiles_from_centerpoints(
	rasterpath: str,
	pointfilepath: str,
	namecol: str = "name",
	df_x: str = "x",
	df_y: str = "y",
	start_end_points: bool = False,
	n_points: Union[int, str] = 100,
	spacing: Union[float, str] = np.sqrt(2),
	swath_number: int = 0,
	swath_spacing: float = 5.,
	band_no: int = 1,
	smooth_first: bool = False,
	method: str = "linear",
	savedir: str = "DEM_profiles",
	view_profiles: bool = False,
	dem_min: float = 0.,
	dem_max: float = 800.,
	verbose: bool = True,
	**pd_kwargs
) -> Tuple[list, list]:
	"""
	Load a .csv containing a list of profile center points, construct
	profile lines from them, and save them in separate .csv files. Also
	returns the created pandas.DataFrame.

	Parameters:
	-----------
		rasterpath: str
			Path to DEM raster (ideally a .tif file or similar).
		pointfilepath: str
			Path to .csv file containing points acting as centers for profile
			lines.
		namecol: str
			Column name in .csv file for the profile names.
		df_x: str
			Name of x coordinate column in .csv file.
		df_y: str
			Name of y coordinate column in .csv file.
		n_points: int
			Number of points to be projected in each direction from centerpoint
			defined by the .csv file in pointfilepath. Total number of points
			will be :math:`2 \cdot n_points + 1`.
			Alternatively, if ``n_points`` is a string, it is interpreted as
			the .csv column containing n_points information for each 
			center point separately.
		spacing: float or str
			Spacing between two consecutive points in the profile.
			Alternatively, if spacing is a string, it is interpreted as
			the .csv column containing spacing information for each
			profile separately.
		swath_number: int
			Number of parallel lines extracted for each profile. Total number
			is :math:`2 \cdot swath_number + 1`.
		swath_spacing: float
			Distance between the swath lines of a profile.
		band_no: int
			Number of band to be interpreted as the DEM in rasterpath file.
			Default is 1.
		smooth_first: bool
			Use a smoothed DEM to calculate the aspect. This will not affect
			the elevation values extracted from the DEM.
		method: str
			Interpolation method for extracting values from the DEM. Default
			is linear interpolation. Can take any values recognized by
			``scipy.interpolate.interpn()``.
		savedir: str
			File path to directory where files are to be saved. If it does not
			exist, the directory is created.
		verbose: bool
			If True, print some status updates while processing.

	Returns:
	--------
		profiles: list
			A list of pandas dataframes that can be forwarded into
			``riserfit.initialize_riser_class()``.
		names: list
			A list of profile names that can be forwarded into
			``riserfit.initialize_riser_class()``.
	"""

	df = pd.read_csv(os.getcwd()+"\\"+pointfilepath, **pd_kwargs)
	#print(df)
	# create directory for storing results
	path = os.getcwd()+"\\"+savedir+"\\"
	if not os.path.exists(path):
		os.makedirs(path)
	
	if start_end_points is False:
		names = df[namecol].to_list()
		xx = pd.to_numeric(df[df_x])
		x = xx.to_numpy()
		yy = pd.to_numeric(df[df_y])
		y = yy.to_numpy()
		points = [(y, x) for x, y in zip(x, y)]

		# get a direction vector associated with each point:
		if verbose: print("Calculating profile azimuths", end="... ", flush=True)
		directions = calculate_aspect(
			rasterpath, points, band_no, smooth_first, "linear"
		)
		print("Done")

	else:
		# check that df has even number of points
		if len(df.index) % 2 != 0:
			raise Exception("Cannot construct profiles from start/end pairs for"
				"uneven number of points in DataFrame.")
		# get x coordinates of starting & endpoints:
		x_start = pd.to_numeric(df[df_x]).to_numpy()[0::2]
		x_end = pd.to_numeric(df[df_x]).to_numpy()[1::2]
		y_start = pd.to_numeric(df[df_y]).to_numpy()[0::2]
		y_end = pd.to_numeric(df[df_y]).to_numpy()[1::2]
		# construct midpoints:
		x = (x_start + x_end) / 2
		y = (y_start + y_end) / 2
		points = [(y, x) for x, y in zip(x, y)] # new midpoints

		# direction vector
		x_dir = x_end - x_start
		y_dir = y_end - y_start
		# normalize
		dir_len = np.sqrt(x_dir**2 + y_dir**2)

		directions = [(y/dir_len[k], x/dir_len[k]) \
			for k, (x, y) in enumerate(zip(x_dir, y_dir))]

		# get names:
		names = df[namecol].to_list()[0::2]

	# prefixes for csv files, ensures correct ordering.

	prof_no = len(names)
	prof_base = "".join(["0"])*len(str(prof_no))
	number = [i for i in range(0, prof_no)]
	prof_name = [prof_base+str(i) for i in number]
	prof_name_cut = [
     	pn[len(str(number[i])):] 
		for i, pn in enumerate(prof_name)
	]

	# check if n_points and/or spacing are str
	# if yes, each spacing and n_points is treated separately
	if type(n_points) == str:
		n_points_array = df[n_points].to_numpy(dtype=int)
	else:
		n_points_array = np.full(len(names), n_points)

	if type(spacing) == str:
		spacing_array = df[spacing].to_numpy()
	else: 
		spacing_array = np.full(len(names), spacing)

	if start_end_points is True:
		# spacing needs is determined by n_points and distance
		# between start and end points
		for i, point in enumerate(points):
			cent_p = np.array(point)
			star_p = np.array([y_start[i], x_start[i]])
			start_end_dist =  np.sqrt(np.sum((cent_p-star_p)**2))
			spacing_array[i] = start_end_dist / (n_points_array[i] + 1)

	# attempt fast execution by bulking all points into one long list...
	try:
     
		# sample raster at all_points
		if verbose: 
			print("Attempting fast point extraction", end="... ", flush=True)
		
		# initialize Constructor_Extractor class:
		ce = Constructor_Extractor(
			points,
			directions,
			start_end_points,
			n_points_array,
			spacing_array,
			swath_number,
			swath_spacing
		)

		ce.build_center_profiles()
		if swath_number > 0:
			ce.build_swath_profiles()
			ce.build_decomposed_point_list()
			all_points = ce.decomposed_swath_line_points
		else:
			ce.build_decomposed_point_list()
			all_points = ce.decomposed_single_line_points
	
		z_profiles = sample_raster_at_points(
			rasterpath=rasterpath,
			rc_points=all_points,
			method=method,
			band_no=band_no
		)	
  
		centerlines = ce.centerlines
  
		if verbose: print("Done")
		ce.reconstruct_elevation_list(z_profiles) # restructure
		z_profiles = ce.profile_z_list

	except MemoryError as E:
     
		if verbose: print("Fast execution failed:", E)
		
		# instead of bulking all points into one long list,
		# call sample_raster_at_points once per profile.
		z_profiles_out = []
		centerlines = []
		for i in range(0, len(points)):
			sys.stdout.write(
       			'\r'+f"Attempting slow point extraction... {i+1}/{len(points)}"
          	)
			ce = Constructor_Extractor(
				[points[i]],
				[directions[i]],
				start_end_points,
				[n_points_array[i]],
				[spacing_array[i]],
				swath_number,
				swath_spacing
			)

			ce.build_center_profiles()
			if swath_number > 0:
				ce.build_swath_profiles()
				ce.build_decomposed_point_list()
				all_points = ce.decomposed_swath_line_points
			else:
				ce.build_decomposed_point_list()
				all_points = ce.decomposed_single_line_points

			z_profiles = sample_raster_at_points(
				rasterpath=rasterpath,
				rc_points=all_points,
				method=method,
				band_no=band_no
			)	

			ce.reconstruct_elevation_list(z_profiles) # restructure
			z_profiles_out.append(ce.profile_z_list[0]) # only one entry??
			centerlines.append(ce.centerlines[0])
   
		z_profiles = z_profiles_out
		if verbose: print("Done")

	# get centerline x, y coordinate, name for each point
	# center profile
	# save to df
	
	profiles_df = []
	if verbose: print(f"Saving profiles", end="... ", flush=True)
	
	for i, name in enumerate(names):

		id_profile = [name+f"-{j}" for j in range(0, 2*n_points_array[i]+1)]
		# x is column direction
		x_profile = [x_col for _, x_col in centerlines[i]]
		y_profile = [y_row for y_row, _ in centerlines[i]]
		z_profile = z_profiles[i]
		
		# mirror if first z > last z
		if z_profile[0] > z_profile[-1]:
			x_profile = x_profile[::-1]
			y_profile = y_profile[::-1]
			z_profile = z_profile[::-1]

		# save to df

		df_dict = {
			"ID": id_profile,
			"x": x_profile,
			"y": y_profile,
			"z": 0  # placeholder, z will be shifted below anyways
		}
		
		profile_df = pd.DataFrame(df_dict)

		# calculate along-profile distance (d)
		profile_df.loc[:, 'dd'] = spacing_array[i]
		profile_df.loc[:, 'd'] = 0.

		profile_df.iloc[:, profile_df.columns.get_loc('d')] = \
			np.cumsum(profile_df['dd'])

		# center the profile: 
		# 1. elevation center, 
		# 2. shift such that z = 0 is
		# at d = 0. 
		# We'll do it in a way such that a point at (0, 0) exists.

		# elevation midpoint:
		z = np.array(z_profile)
		z_mid = (z.max() + z.min()) / 2
		id_mid = np.where(z<z_mid)[0][-1]
		z_mid = z[id_mid]
		z = z - z_mid
		profile_df["z"] = z

		# shift distance such that z_mid is located at d = 0:
		d_mid = profile_df.iloc[id_mid]["d"]
		d = profile_df.loc[:, "d"] - d_mid
		profile_df.loc[:, "d"] = d

		profiles_df.append(profile_df)

		# save dfs to .csv (NOTE: not in splitprofiles/centerprofiles anymore!!)
		profile_df.to_csv(path+f"{prof_name_cut[i]}_{name}.csv")
	
	print("Done")

	if view_profiles:

		# make a quick and dirty map of dem
		raster = rio.open(os.getcwd()+"\\"+rasterpath)
		dem = raster.read(band_no)
		raster_bounds = raster.bounds
		raster_bounds = [
			raster_bounds[0],
			raster_bounds[2],
			raster_bounds[1],
			raster_bounds[3]
		]
		plt.imshow(dem, extent=raster_bounds, vmin=dem_min, vmax=dem_max)
		for i, profile in enumerate(profiles_df):
			plt.plot(profile["x"], profile["y"], c="black")
		plt.show()
	return (profiles_df, names)


# function that returns the direction of steepest slope for each point specified
# on a DEM

def calculate_aspect(
	rasterpath: str,
	rc_points: list[tuple],
	band_no: int = 1,
	smooth_first: bool = False,
	method: str = "linear",
	verbose: bool = True,
	**rio_open_params
) -> list[tuple]:

	"""
	Calculate the local aspect of a list of points given in row-column format
	:math:`(y, x)` on a DEM in the same coordinate system.

	Parameters:
	-----------
		rasterpath: str
			Path to the DEM.
		rc_points: list[tuple]
			List of points stored as :math:`(y, x)` tuples.
		band_no: int
			Band of the input raster to be interpreted as the DEM.
		smooth_first: bool
			Option to perform the azimuth calculation on a smoothed DEM.
			This will generally lead to better results, unless the topography
			is very complex.
		method: str
			Method used to calculate and interpolate the elevation values at the
			wanted points. Can take any value permitted by
			``scipy.interpolate.interpn()``.

	Returns:
	--------
		azimuths: list[tuple]
			List of :math:`(y, x)` tuples representing the vector pointing in the
			direction of steepest descent, in the same order as ``rc_points``.
	"""
	raster = rio.open(os.getcwd()+"\\"+rasterpath, **rio_open_params)
	dem = raster.read(band_no)
	# get spacings along x, y
	row_id = [i for i in range(0, dem.shape[0])]
	col_id = [i for i in range(0, dem.shape[1])]

	# get y coordinate values for rows
	row_coords = rio.transform.xy(
		raster.transform,
		row_id,
		np.zeros(len(row_id), dtype="int")
	)[1]

	# get x coordinate values for cols
	col_coords = rio.transform.xy(
		raster.transform,
		np.zeros(len(col_id), dtype="int"),
		col_id
	)[0]
	raster_coordinates = (row_coords, col_coords)
	# calculate gradient (aspect not needed in degree, rather a vector is
	# sufficient).

	if smooth_first is True:
		if verbose: print("Smoothing DEM for azimuth calculations", end="... ", flush=True)
		dem = gaussian_filter(dem, [4, 4], mode="constant")
	drow, dcol = np.gradient(dem, row_coords, col_coords)

	drow_int = interpn(
		raster_coordinates,
		drow,
		rc_points,
		method
	)

	dcol_int = interpn(
		raster_coordinates,
		dcol,
		rc_points,
		method
	)

	# row, column format
	az = [(y, x) for x, y in zip(dcol_int, drow_int)]
	# normalize to len = 1
	len_az = [np.sqrt(y**2+x**2) for y, x in az]
	norm_az = [(vect[0]/len_az[i], vect[1]/len_az[i]) \
		for i, vect in enumerate(az)]

	return norm_az

# function that takes a center point & direction vector and extrapolates
# n points in both directions with even spacing
def extrapolate_line_from_point(
	centerpoint: tuple,
	direction: tuple,
	n_points: int = 100,
	spacing: float = 1.5
) -> list[tuple]:
	"""
	Returns an array of points defining a profile line that can be fed into
	sample_raster_at_points(). Input should be in row-column format (first
	entry in tuple denotes the row or y position, second entry the col or x
	position).

	Parameters:
	-----------
		centerpoint: tuple
			The point of origin. A profile will be constructed with n_points in
			the direction of vector and :math:`-1\cdot vector` for a total of :math:`2\cdot n_points+1`
			points.
		direction: tuple
			A tuple defining the orientation of the profile line.
			The line is constructed as :math:`new_points = centerpoint+a\cdot direction`.
		n_points: int
			Number of point to be returned in each direction of the centerpoint.
		spacing: float
			Spacing between any two neighbouring points.

	Returns:
	--------
		rc_points: list[tuple]
			An array of point tuples ordered along the first dimension.
	"""
	direction = np.array(direction)
	centerpoint = np.array(centerpoint)

	# normalize direction vector to length spacing:
	direction = direction/(np.sqrt(np.sum(direction**2)))*spacing
	# points in direction:
	# first point is at centerpoint+direction, etc.
	points_1 = [tuple(centerpoint+direction*i) for i in range(1, n_points+1)]

	# points in -1*direction:
	points_2 = [tuple(centerpoint-direction*i) for i in range(1, n_points+1)]
	# merge and sort by first value
	prof_points = points_1 + [tuple(centerpoint)] + points_2
	#print(prof_points)
	prof_points = sorted(prof_points)
	#
	return prof_points

# 2D raster interpolator with any method
def sample_raster_at_points(
	rasterpath: str,
	rc_points: np.array,
	method: str = "linear",
	band_no: int = 1,
	rio_open_dict: dict = {}
) -> np.ndarray:
	"""
	This function samples a raster that can be read by rasterio at the required
	points using a user-defined interpolation method. Points should be given in
	(row, column) / (y, x) format.

	Parameters:
	-----------
		rasterpath: str
			Relative path from current working directory (os.getcwd()) to the
			raster file containing a DEM.
		rc_points: np.array
			1D array containing y,x coordinate tuples of the points where the
			raster should be sampled at.
		method: str
			Interpolation method used by scipy.interpolate.interpn().
			Default is linear interpolation.
		band_no: int
			Specifies the band of the raster that will be read as the DEM.
		rio_open_dict: dict
			Any parameters relating to rasterio.open()

	Returns:
	--------
		rc_int: np.ndarray
			List of elevation values corresponding to the points in rc_points.
	"""

	# get coordinates of rio_raster rows, cols:
	raster = rio.open(os.getcwd()+"\\"+rasterpath, **rio_open_dict)
	dem = raster.read(band_no)
	row_id = [i for i in range(0, dem.shape[0])]
	col_id = [i for i in range(0, dem.shape[1])]

	# get y coordinate values for rows
	row_coords = rio.transform.xy(
		raster.transform,
		row_id,
		np.zeros(len(row_id), dtype="int")
	)[1]

	# get x coordinate values for cols
	col_coords = rio.transform.xy(
		raster.transform,
		np.zeros(len(col_id), dtype="int"),
		col_id
	)[0]
	
	raster_coordinates = (row_coords, col_coords)

	rc_int = interpn(
		raster_coordinates,
		dem,
		rc_points,
		method
	)
	return rc_int

def load_profile_csvs(
	filepath: str,
	pandas_csv_dir: dir = {}
) -> Tuple[list, list]:
	"""
	load existing profile .csv files located in a splitprofiles\\centerprofiles
	folder.

	Parameters:
	-----------
		filepath: str
			Path from current working directory to the directory containing
			the .csv files to be loaded.
		pandas_csv_dir: dir
			Any arguments passed to ``pandas.read_csv()``.

	Returns:
	--------
		profiles: list
			List of pd.DataFrame containing the profiles.
		names: list
			List of names corresponding to the profiles.
	"""

	if os.path.isdir(os.getcwd()+"\\"+filepath) is False:
		raise Exception(f"{os.getcwd()}\\{filepath} does not exist.")
	
	loadpath = f"{os.getcwd()}\\{filepath}\\"
	files = [
    	f for f in os.listdir(loadpath)
    	if os.path.isfile(os.path.join(loadpath, f))
	]
	names = [f[:-4] for f in files]
	profiles = [pd.read_csv(loadpath+f, **pandas_csv_dir) for f in files]

	return (profiles, names)
