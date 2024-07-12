# type hints
from __future__ import annotations # necessary for some typing shenanigans
from numpy.typing import ArrayLike
from typing import Union, Tuple, List
from typing_extensions import Self # pre python 3.11

#system stuff
import os, warnings, sys, traceback

# data analysis, managing and calculations
import numpy as np
import pandas as pd
import rasterio as rio

# imports for (non-)linear diffusion fitting
from scipy.optimize._optimize import OptimizeResult
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# for plotting stuff
import matplotlib.pyplot as plt
import matplotlib as mpl
# may change backend to cairo in future, for better plots?
from matplotlib.backends.backend_pdf import PdfPages

# library-internal functions
from .diffusion import nonlin_diff_perron2011
from .dem import *
from .riser_maths import *
# from .inversion import *
from .inversion import (
    _compute_misfit_for_optimization, 
    _linear_kt_uncertainty_mse,
    _nonlin_invert_uncertainty,
    _lin_invert_uncertainty    
)

#########################################################################
## Part 1: Dealing with files, file management, setting up Riser class ##
#########################################################################

# Function to open a collection of profiles

def read_profile_files(
    filepath: str,
    infiles: list[str],
    profilenames: list[str],
    identifier: str,
    df_x: str = "x_UTM",
    df_y: str = "y_UTM",
    pandas_csv_dict: dict = {}
) -> Tuple[list, list]:

    """
    read_profile_files: Function to open and prepare data files for terrace
    riser analysis. Files should be provided in .csv format and contain
    x, y coordinates in meters (e.g. UTM), and elevation. Multiple riser
    profiles may be stored in a single .csv file, as long as there is an
    identifier column with unique names for each individual profile.
    (E.g., all points are labelled as riser1-001, 
    riser1-002, ..., riser2-001, ... The identifiers should then be "riser1",
    "riser2", etc.)

    Parameters:
    -----------
        filepath: str
            Path to directory where data files with profile information are
            stored.
        infiles: list[str]
            List of .csv files containing profiles to be loaded.
        profilenames: list[str]
            List of profile names to be searched for in the .csv files.
            All rows that are assigned to a profile must contain the profile
            name in the identifier column.
        identifier: str
            Name of the column that contains the name of a profile listed
            in profilenames.
        df_x: str
            Name of column in .csv file where x coordinates of the profiles
            are stored.
        df_y: str
            Name of column in .csv file where y coordinates of the profiles
            are stored.
        pandas_csv_dict: dir
            Any arguments to be passed on to pandas.read_csv()

    Returns:
    --------
        profiles: list
            List of pandas dataframes containing separate profiles for which
            data was found.
        profilenames: list[str]
            List containing all names of profiles for which data was found.

    """

    # create a df for each infile

    files = []

    for i, infile in enumerate(infiles):
        df = pd.read_csv(
            f"{os.getcwd()}\\{filepath}\\{infile}", 
            **pandas_csv_dict
        )

        # use the xy_distances function
        x = df[df_x].to_list()
        y = df[df_y].to_list()
        d, dd = xy_distances(x, y)
        df["d"] = d 
        df["dd"] = dd
        files.append(df)

    profiles = []  # store separate profiles here
    profile_not_found = np.full(len(profilenames), True, dtype="bool")

    for i, name in enumerate(profilenames):
        # for each infile, check if name is in the identifier column
        for file in files:
            if any(file[identifier].str.contains(name)):
                profiles.append(file.loc[file[identifier].apply(lambda x: name in x) == True].copy())
                profile_not_found[i] = False

    profilenames = np.array(profilenames)  # convert to np array for easier indexing
    if any(profile_not_found):
        warnings.warn(f"WARNING: Could not find points for profiles {profilenames[profile_not_found]}.")

    return (profiles, profilenames[~profile_not_found])


def _split_profile_GUI(profile, df_x="x_UTM", df_y="y_UTM"):

    """
    This function is internal and should only be called by split_elevation_profiles().
    """

    indexlist = []

    def onpick(event, df_x=df_x, df_y=df_y):
        ind = event.ind[0]
        if ind in indexlist:
            indexlist.remove(ind)
            print(f"Removed point with ID {ind}.")
            plt.scatter(profile.iloc[ind][df_x], profile.iloc[ind][df_y],
                        c="blue")
            plt.show()
        else:
            indexlist.append(ind)
            print("Point ID:", ind,
                  f"| {df_x}:", profile.iloc[ind][df_x],
                  f"| {df_y}:", profile.iloc[ind][df_y])
            plt.scatter(profile.iloc[ind][df_x], profile.iloc[ind][df_y],
                        c="red")
            plt.show()

    fg, ax = plt.subplots()
    ax.scatter(profile[df_x], profile[df_y], marker="+", picker=True)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title("Select two points by clicking.")
    fg.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    print("Selected indices:", indexlist)
    return indexlist

def split_elevation_profiles(
    profiles: list,
    profilenames: list[str],
    savedirpath: str,
    df_x: str = "x_UTM",
    df_y: str = "y_UTM",
    df_z: str = "z",
    verbose: bool = True
) -> Tuple[list, list]:
    """
    Split elevation profiles into individual riser profiles using
    an interactive GUI. Process is saved along the way, even if the
    script execution is interrupted at some point. User can decide
    whether to split a profile via console inpute, and then select
    the points between which a profile should be exctracted in a 
    plot window.

    Parameters:
    -----------
        profiles: list
            List of input elevation profile dfs.
        profilenames: list[str]
            List of input elevation profile names.
        savedirpath: str
            Path to subdirectory where resulting .csv files
            are to be saved.
        df_x: str
            Name of df column containing x coordinate values.
        df_y: str
            Name of df column containing y coordinate values
        df_z: str
            Name of df column containing elevation values
        verbose: bool
            Option to print some additional status updates 
            to console.

    Returns:
    --------
        output_df_collection: list
            List of dfs for each individual riser profile.
        output_name_collection: list
            List of unique names for each riser profile.

    """

    if len(profiles) != len(profilenames):
        raise Exception("Profiles and profilenames not of equal length.")
    
    # check or create path to saved profiles
    datapath = f"{os.getcwd()}\\{savedirpath}\\"
    if not os.path.isdir(datapath):
        try: 
            os.mkdir(datapath)
        except OSError:
            print(f"Failed to create {datapath}.")
            sys.exit(1)

    # gather all .csv files in the subdirectory
    # check names that exist as .csv files and are in profilenames
    # continue splitting profiles that are not already
    # saved as .csv
    dirfiles = os.listdir(datapath)
    dirfiles = [f for f in dirfiles if os.path.isfile(datapath+f)]
    # which profile names are in directory?
    dir_file_names = "\t".join(dirfiles)+"r4nd0m" # avoid empty str
    done = [True if n in dir_file_names else False for n in profilenames]
    notdone = [not elem for elem in done]

    profilenames = np.array(profilenames)
    if verbose:
        done_profiles = ", ".join(profilenames[done])
        notdone_profiles = ", ".join(profilenames[notdone])
        print(f"Found split profiles: {done_profiles}")
        print(f"Proceeding with: {notdone_profiles}")

    # select profiles that need to be split...
    to_split_profiles = \
        [p for i, p in enumerate(profiles) if notdone[i]]
    to_split_names = profilenames[notdone]

    split_profiles = []
    split_profile_names = []
    n = len(to_split_profiles)

    for i, df in enumerate(to_split_profiles):

        # show a plot with x, y coordinates
        # and d, z to allow user decision...
        if verbose:
            print(f"Inspecting {to_split_names[i]} [{i+1}/{n}]")

        fg, axs = plt.subplots(1, 2, figsize=(8, 5))
        axs[0].scatter(df[df_x], df[df_y], marker="+")
        axs[0].set_title("Close to proceed.")
        axs[0].set_xlabel("Easting [m]")
        axs[0].set_ylabel("Northing [m]")
        axs[1].scatter(df['d'], df[df_z], marker="+")
        axs[1].set_xlabel("Along-profile distance [m]")
        axs[1].set_ylabel("Relative elevation [m]")
        axs[1].set_title(to_split_names[i])
        plt.show()

        split_yn = \
            input(f"Split profile {to_split_names[i]}? [y/n] ")

        # easy case first... "n" -> no action and save
        if split_yn.lower() == "n":
            
            split_profiles.append(df)
            split_profile_names.append(to_split_names[i])
            df.to_csv(f"{datapath}{to_split_names[i]}.csv")
            if verbose:
                print(f"{to_split_names[i]} processed. Proceeding to next profile.")

        elif split_yn.lower() == "y":
            
            counter = 0 # may be split many times...
            
            while split_yn.lower() == "y":
                
                ind = _split_profile_GUI(df, df_x=df_x, df_y=df_y)
                while len(ind) < 2: # ensure correct point picking
                    print("Selected less than two points, please retry!")
                    ind = _split_profile_GUI(df, df_x=df_x, df_y=df_y)
                if len(ind) > 2:
                    print("Selected more than two points, choosing endpoints!")
                
                # generate df with start and stop ind and save 
                df_split = df.iloc[min(ind):max(ind)].copy()
                split_profiles.append(df_split)
                split_profile_names.append(f"{to_split_names[i]}_{counter}")
                df_split.to_csv(f"{datapath}{split_profile_names[-1]}.csv")

                # again?
                split_yn = input(f"Split stub {to_split_names[i]} again? [y/n] ")
                counter += 1
        else: 
            raise Exception("Console input must be y or n!")

        if verbose: print(f"Profile {to_split_names[i]} processed.")
    
    if verbose: print("All profiles processed.")

    # collect profiles for function return
    # load from datapath...
    split_dfs = os.listdir(datapath)
    split_dfs = [f for f in split_dfs if \
        os.path.isfile(f"{datapath}{f}")] # all csv files found.
    
    # select file names present in profilenames
    output_file_collection = []
    for file in split_dfs:
        add_to_file_output = [True for n in profilenames if n in file]
        if len(add_to_file_output) > 0:
            output_file_collection.append(file) 

    output_df_collection = [pd.read_csv(f"{datapath}{f}") \
        for f in output_file_collection]
    output_name_collection = [n[:-4] for n in output_file_collection]

    return (output_df_collection, output_name_collection)


# Function to reproject profiles onto straight line, defined by lst sq fit
def reproject_profiles(
    profiles: list,
    df_x: str = "x_UTM",
    df_y: str = "y_UTM"
) -> list:

    """
    For a list of pandas dfs containing gps profiles,
    performs least square reprojection and re-calculates distances.
    This function should only be used for profiles of individual
    risers!

    Parameters:
    -----------
        profiles: list
            List of pandas dfs containing riser profiles.
        df_x: str
            Name of the df x coordinate column.
        df_y: str
            Name of the df y coordinate column.

    Returns:
    --------
        df_out: list
            Updated pandas dfs with reprojected profiles.
    """

    dfs_out = []

    for i, df in enumerate(profiles):
        x = df[df_x].to_list()
        y = df[df_y].to_list()

        x_new, y_new, d_new, dd_new = \
            least_square_reproject(x, y)
        df[df_x] = x_new
        df[df_y] = y_new
        df["d"] = d_new
        df["dd"] = dd_new
        dfs_out.append(df)

    return dfs_out


def center_elevation_profiles(
    profiles: list,
    df_z: str = "z"
) -> list:
    """
    Automatically centers a profile close to the riser midpoint.
    Uses the difference in maximum and minimum elevation to
    find the midpoint height, and then shifts in d direction
    such that this point is at d=0. Any additional calibration
    of the profile should be done using 
    Riser.compute_best_linear_diffusion_fit() and
    Riser.apply_d_z_offsets().

    Parameters:
    -----------
        profiles: list
            List of pandas dfs with profile information.
        df_z: str
            Name of df column containing elevation information.

    Returns:
    --------
        out_profiles: list
            Profile dfs with updated distances and elevations.

    """

    out_profiles = []
    for df in profiles:

        z = df[df_z].to_numpy()
        z_mid = (z.max()+z.min())/2
        id_mid = np.where(z<z_mid)[0][-1]

        # shift in z
        z_mid = z[id_mid]
        z = z - z_mid
        df[df_z] = z

        # shift in d
        d = df["d"].to_numpy()
        d = d - d[id_mid]
        df["d"] = d

        out_profiles.append(df)

    return out_profiles


def initialize_riser_class(
    profiles: list,
    names: list,
    df_x: str = "x_UTM",
    df_y: str = "y_UTM",
    df_z: str = "z",
    identifier: str = "Riser"
) -> Riser:
    """
    Function to construct the preprocessed profiles into
    objects of the Riser class. Always returns one Riser class object that
    stores all information in lists.

    Parameters:
    -----------
        profiles: list
            Profiles to be initialized as Riser objects. Should be pandas
            dataframes.
        names: list
            Names of profiles.
        df_x: str
            Name of column in .csv files where x coordinates of the profiles
            are stored.
        df_y: str
            Name of column in .csv files where y coordinates of the profiles
            are stored.
        df_z: str
            Name of column in .csv files where z coordinates of the profiles
            are stored.

    Returns:
    --------
    Riser:
        Riser class object.
    """

    if len(names) != len(profiles):
        raise Exception(f"Number of provided names does not"
                        f" equal number of provided profiles: {len(names)}, {len(profiles)}")

    x, y, z, d, name = [], [], [], [], []

    for i, profile in enumerate(profiles):
        x.append(profile[df_x].to_numpy())
        y.append(profile[df_y].to_numpy())
        z.append(profile[df_z].to_numpy())
        d.append(profile["d"].to_numpy())
        name.append(names[i])

    return Riser(x=x, y=y, z=z, d=d, name=name, identifier=identifier)

###########################################################################
## Part 2: Dealing with risers, misfits, and the main Riser class itself ##
###########################################################################


class Riser:
    """
    Riser: class containing geometrical information about risers (x, y, z, and
    along-profile distance). Data is stored in lists to allow for an arbitrary
    amount of risers to be processed at once. This class contains various 
    methods used for calculating riser ages.

    Attributes:
    -----------
        x: list[np.array]
            Contains x coordinates for each profile.
        y: list[np.array]
            Contains y coordinates for each profile.
        z: list[np.array]
            Contains z coordinates for each profile.
        d: list[np.array]
            Contains along-profile distances for each profile.
        name: list[str]
            Contains names for each profile.
        identifier: str
            Contains string used to identify PDFs generated by class instance.
    """

    def __init__(
        self,
        x: list[np.ndarray],
        y: list[np.ndarray],
        z: list[np.ndarray],  # elevations
        d: list[np.ndarray],  # along profile distances
        name: list[str],  # profile name
        identifier: str # identify string
    ):
        """
        Initialises a Riser instance and pre-defines all attributes as
        empty lists. This is done to achieve maximum compatibility
        between different instances.
        
        Parameters:
        -----------
            x: list[np.ndarray]
                x coordinates of riser profiles.
            y: list[np.ndarray]
                y coordinates of riser profiles.
            z: list[np.ndarray]
                Elevations of riser profiles.
            d: list[np.ndarray]
                Between-point distances of riser profiles.
            name: list[str]
                Profile names.
            identifier: str
                Identifier of the profiles.
        
        Returns:
        --------
            None
            
        """
        # geometric properties and name tag(s)
        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.name = name
        self.identifier = identifier

        #############################################################
        # collection of class attributes that are assigned later on #
        #############################################################

        # by compute_symmetry_index()
        self.symmetry_index = []
        """Attribute in which compute_symmetry_index() stores results."""

        # by compute_gradients()
        self.gradient = []  # numerical central-difference derivate along d
        """Attribute in which compute_gradients() stores results."""
        
        # by compute_azimuth()
        self.azimuth = [] # in rad or grad, user defined
        """Attribute in which compute_azimuth() stores results."""
        
        # by compute_best_linear_diffusion_fit()
        self.optimize_summary = []  # output of scipy.optimize.minimize
        """Attribute in which compute_best_linear_diffusion_fit() stores optimisation result."""
        self.best_misfit = []  # misfit for best param combination
        """Attribute in which compute_best_linear_diffusion_fit() stores RMSE."""
        self.best_kt = []  # diffusion age
        """Attribute in which compute_best_linear_diffusion_fit() stores kt."""
        self.best_a = []  # half the initial riser height
        """Attribute in which compute_best_linear_diffusion_fit() stores best riser height."""
        self.best_b = []  # far field gradient
        """Attribute in which compute_best_linear_diffusion_fit() stores best far-field gradient."""
        self.best_theta = []  # not optimized for, kept constant
        """Attribute in which compute_best_linear_diffusion_fit() stores best intitial slope."""
        self.best_d_off = []  # optimal offset in horizontal direction
        """Attribute in which compute_best_linear_diffusion_fit() stores d offset."""
        self.best_z_off = []  # optimal offset in vertical direction
        """Attribute in which compute_best_linear_diffusion_fit() stores z offset."""
        self.best_midpoint_grad = [] # riser midpoint slope for the best fit
        """Attribute in which compute_best_linear_diffusion_fit() stores gradient at midpoint."""
        
        # by compute_best_nonlinear_diffusion_fit()
        self.nonlin_best_t = []  # best fit age
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores best t."""
        self.nonlin_opt_sum = []  # output of minimize()
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores optimisation result."""
        self.nonlin_best_z = []  # save best fit z, because not easy to re-calculate
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores best-fit z matching self.d."""
        self.nonlin_z_matrix = [] # save nonlin profile ast all time steps
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores z profiles at time steps dt."""
        self.nonlin_t_times = [] # saves all timesteps of nonlin profile diffusion
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores all computed time steps."""
        self.nonlin_d = [] # distances used by nonlin diffusion scheme
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores optimisation result."""
        self.nonlin_best_mf = [] # minimum misfit value for nonlin fit
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores RMSE."""
        self.nonlin_dt = [] # contains dt used for diffusion
        """Attribute in which compute_best_nonlinear_diffusion_fit() stores used dt."""
        
        # by calculate_nonlin_t_uncertainty()
        self.nonlin_upper_t = []
        """Attribute in which calculate_nonlin_t_uncertainty() stores upper bounds."""
        self.nonlin_lower_t = []
        """Attribute in which calculate_nonlin_t_uncertainty() stores lower bounds."""

        # by jackknife_linear_diffusion_fit()
        # these lists should be nested: 
        # [[[],[],[],...],[[],[],[],...],...]
        self.jack_d = [] # d values reduced by d samples
        """Attribute in which jackknife_linear_diffusion_fit() stores distances."""
        self.jack_z = [] # z values reduced by d samples
        """Attribute in which jackknife_linear_diffusion_fit() stores elevations."""
        self.jack_kt = [] # kt for each reduced profile
        """Attribute in which jackknife_linear_diffusion_fit() stores kt."""
        self.jack_a = [] # a for each reduced profile
        """Attribute in which jackknife_linear_diffusion_fit() stores riser heights."""
        self.jack_b = [] # b for each reduced profile
        """Attribute in which jackknife_linear_diffusion_fit() stores far-field slopes."""
        self.jack_d_off = [] # slightly nsfw, d offset for red. profile
        """Attribute in which jackknife_linear_diffusion_fit() stores d offsets."""
        self.jack_z_off = [] # z offset for reduced profile
        """Attribute in which jackknife_linear_diffusion_fit() stores z offsets."""
        self.jack_optimize_summary = [] # summary of scipy optimization
        """Attribute in which jackknife_linear_diffusion_fit() stores optimisation summaries."""
        self.jack_misfit = [] # best misfit values
        """Attribute in which jackknife_linear_diffusion_fit() stores RMSE."""
        self.jack_midpoint_gradient = [] # midpoint gradients
        """Attribute in which jackknife_linear_diffusion_fit() stores midpoint gradients."""

        # by jackknife_nonlinear_diffusion_fit()
        self.nonlin_jack_best_t = [] # best t nonlinear jackknife
        """Attribute in which jackknife_nonlinear_diffusion_fit() stores t."""
        # by add_cn_ages()
        self.cn_age = []  # cosmogenic nuclide ages, if available
        """Attribute in which add_cn_ages() stores ages."""
        self.cn_age_sigma = []
        """Attribute in which cn_age_sigma() stores uncertainty."""
        self.cn_age_reliability = []  # string describing the reliability of an age
        """Attribute in which cn_age_sigma() stores a reliability indicator."""
        
        # by add_cn_terrace_age()
        self.terrace_age = []
        """Attribute in which add_cn_terrace_age() stores ages."""
        self.terrace_age_sigma = []
        """Attribute in which add_cn_terrace_age() stores uncertainty."""

        # by add_terrace_generation()
        self.terrace = []
        """Attribute in which add_terrace_generation() stores the generation."""

        # by calculate_kt_uncertainty()
        self.upper_kt = []  # upper uncertainty bound for kt
        """Attribute in which calculate_kt_uncertainty() stores an upper bound."""
        self.lower_kt = []  # lower uncertainty bound for kt
        """Attribute in which calculate_kt_uncertainty() stores a lower bound."""

        # by extract_profile_elevation()
        self.profile_elevation = [] # profile elevation from DEM or REM
        """Attribute in which extract_profile_elevation() stores an absolute elevation value."""

        # by calculate_upstream_distance()
        self.upstream_distance = [] # distance from river base
        """Attribute in which calculate_upstream_distance() stores upstream_distance."""

    def extract_subset(
        self,
        profiles = []
    ) -> Riser:
        """
        extract_subset: Function to construct a new Riser class object with
        a subset of profiles from the original class instance.  All attributes of the original
        instance are preserved.
        
        The extraction process is limited to attributes of len(self.name).

        Parameters:
        -----------
            profiles: list of int or str
                List containing either the index positions or names of profiles
                to be subset.

        Returns:
        --------
            new_riser: Riser
                A new Riser instance.

        """

        # return empty instance if no profiles are specified
        if profiles == []: return Riser([], [], [], [], [], self.identifier)
        
        # if entries are str, treat as profile names
        if type(profiles[0]) == str: 
            indices = [self.name.index(p) for p in profiles]
        else: # expect ints here
            indices = profiles

        # create a new, empty instance of riser class:
        new_riser = Riser([], [], [], [], [], self.identifier)

        # get all other attributes:
        attribute_keys = self.__dict__.keys()

        for key in attribute_keys:
            attribute = getattr(self, key)
            if type(attribute) == list and len(attribute) == len(self.name):
                attribute_subset = [attribute[i] for i in indices]
                setattr(new_riser, key, attribute_subset)

        return new_riser

    def compute_azimuth(
        self,
        rad: bool = True 
    ) -> Self:
        """
        Calculates the profile orientation relative to north. 

        Parameters:
        -----------
            rad: bool
                Angles are returned in rad. If False, returned in 
                degrees.
        
        Returns:
        --------
            self: Self
                The Riser instance.
        """
        azs = []
        for i, name in enumerate(self.name):
            n = np.array([0, 1])
            p = np.array([
                self.x[i][-1]-self.x[i][0],
                self.y[i][-1]-self.y[i][0]
            ])

            dot = n[0]*p[0] + n[1]*p[1] # Dot product between [x1, y1] and [x2, y2]
            det = n[0]*p[1] - n[1]*p[0] # Determinant
            angle = np.arctan2(det, dot)  
            if rad is False:
                angle *= 180/np.pi 
            azs.append(angle)

        self.azimuth = azs
        
        return self


    def flip_profile_orientation(
        self
    ) -> Self:
        """
        Function to ensure that each d and z are ordered such
        that d[0] < d[-1] and z[0] < z[-1]. Also reverses the ordering
        in x and y. This is important for symmetry calculations and
        general data fitting.

        Parameters:
        -----------
            None
            
        Returns:
        --------
            self: Self
                The Riser instance.
        """

        # check if z[0] < z[-1]. If yes, done. If not, flip.
        for i, name in enumerate(self.name):

            if self.z[i][0] > self.z[i][-1]:

                # x, y might be empty in some cases...
                xy_empty = False if len(self.x) > 0 else True
                # flip x, y, z, (d)
                if not xy_empty:
                    self.x[i] = self.x[i][::-1]
                    self.y[i] = self.y[i][::-1]
                self.z[i] = self.z[i][::-1]

                # calculate d from scratch:
                d_new, _ = xy_distances(
                    self.x[i], self.y[i]
                )

                # re-center z, just to be sure...
                z_mid = (self.z[i].max()+self.z[i].min())/2
                id_mid = np.where(self.z[i]<z_mid)[0][-1]

                # shift in z
                z_mid = self.z[i][id_mid]
                
                self.z[i] = self.z[i] - z_mid
                
                # shift in d
                self.d[i] = d_new - d_new[id_mid]
            
        return self

    def compute_symmetry_index(
        self,
        summarypdf: bool = False,
        verbose: bool = True
    ) -> Self:
        """
        Calculates the RMSE between the lower and upper "legs" of the
        riser. The midpoint, or point of rotation, is defined at d=0.
        
        Parameters:
        -----------
            summarypdf: bool
                Whether to output a PDF containing symmetry index results.
            verbose: bool
                Whether to print symmetry indices to the console.
        
        Returns:
        --------
            self: Self
                The Riser instance.
        """

        if summarypdf is True:
            p = PdfPages(os.getcwd()+f"\\{self.identifier}_symmetry_summary.pdf")

        symm_index = []
        if verbose and summarypdf:
            print(
                f"\nPrinting symmetry summary PDF to {os.getcwd()}", 
                end="... ", 
                flush=True
            )
        for i, d in enumerate(self.d):
            d_neg = -d[d<0]
            z_neg = -self.z[i][d<0]


            intfun = interp1d(d_neg, z_neg, kind="cubic")
            d_inds = \
                [True if dist >= d_neg.min() and dist <= d_neg.max() else False\
                for dist in d]
            z_int = intfun(d[d_inds])
            a = self.best_a[i]
            #a = 1
            error = \
                (1/a)*np.sqrt((np.sum((self.z[i][d_inds]-z_int)**2)) / \
                len(z_int))

            if summarypdf is True:

                fg, ax = plt.subplots(1,2)
                ax[0].plot(d, self.z[i])
                ax[0].set_title(self.name[i])
                ax[1].plot(d_neg, z_neg, c="red", label="rotated")
                ax[1].plot(d[d_inds], self.z[i][d_inds], c="blue",
                    label="upper")
                #ax[0].plot(d[d_inds], z_int, c="black", label="int",
                #    linestyle="dashed")
                ax[1].set_title(f"SE = {error:.4f}")
                ax[1].legend()
                fg.savefig(p, format="pdf")
                plt.close()

            symm_index.append(error)

        self.symmetry_index = symm_index
        if summarypdf:
            p.close()
            if verbose: print("Done")
        
        return self

    def compute_gradients(
        self,
        summarypdf: bool = False,
        verbose: bool = True
    ) -> Self:

        """
        compute_gradients: Function to compute the numerical gradient for each
        profile point based on the along-profile distance d and elevation z.
        The centered derivative is calculated for all points except for the
        first and last one. For the latter, the forwards/backwards derivative is
        calculated instead.

        Parameters:
        -----------
            summarypdf: bool
                Option to create a PDF in the working directory.
            verbose: bool
                Option to output a message when creating PDF.

        Returns:
        --------
            self: Self
                The Riser instance.
        """

        grads = []

        for i, x_coord in enumerate(self.d):
            y_coord = self.z[i]
            center = (y_coord[2:] - y_coord[:-2]) / \
                     (x_coord[1:-1] - x_coord[:-2] + x_coord[2:] - x_coord[1:-1])
            first = (y_coord[1] - y_coord[0]) / (x_coord[1] - x_coord[0])
            last = (y_coord[-1] - y_coord[-2]) / (x_coord[-1] - x_coord[-2])
            grad = np.concatenate(([first], center, [last]))
            grads.append(grad)

        self.gradient = grads

        # create a summary pdf if wanted

        if summarypdf is True:

            if verbose is True:
                print("\nCreating gradient summary PDF", end="... ")

            p = PdfPages(os.getcwd()+f"\\{self.identifier}_gradient_summary.pdf")

            for i, name in enumerate(self.name):

                fg, ax = plt.subplots()
                ax.scatter(self.d[i], self.gradient[i], c="black")
                ax.set_title(name)
                # compute and plot smooth gradient (10% window width)
                n = int(len(self.d[i])*0.1)

                smooth_grad = \
                    np.convolve(self.gradient[i], np.ones(n)/n, "same")
                ax.plot(self.d[i], smooth_grad, c="red")
                ax.set_ylabel("Gradient [m/m]")
                ax.set_xlabel("Along-profile distance [m]")
                ax.set_ylim([-0.1, 0.8])
                fg.savefig(p, format="pdf")
                plt.close()

            p.close()

            if verbose is True:
                print("Done")
        
        return self

    def apply_d_z_offsets(self) -> Self:
        """
        Function to recalculate d, z using the best fit
        offsets determined by the linear diffusion fit.
        
        Parameters:
        -----------
            None
            
        Returns:
        --------
            None
            
        """

        # check if best_d_off, best_z_off are present:

        checks = [self.best_d_off == [], self.best_z_off == []]
        if any(checks):
            raise Exception("self.best_d_off or self.best_z_off are empty.")
        else:
            for i, dists in enumerate(self.d):
                self.d[i] = dists + self.best_d_off[i]
                self.z[i] = self.z[i] + self.best_z_off[i]
        
        return self


    def mirror_upper_profile_half(
        self,
        apply_to_selection: list = None,
        symmetry_threshold: float = 4.5,
        summarypdf: bool = False,
        verbose: bool = True
    ) -> Self:
        """
        This method takes the upper half of each riser profile
        (defined by d>0) and rotates it in such a way that it 
        replaces the lower half of the profile. This is
        equivalent to a rotation of 180° around the coordinate
        origin. By default, it is applied to all profiles with a 
        symmetry index greater than 4.5.

        Parameters:
        -----------
            apply_to_selection: list
                If not None (default), a list of length equal to number
                of profiles containing True/False entries.
                If entry is True, the profile corresponding to the index
                is mirrored.
            symmetry_threshold: float
                All profiles with a SE greater than the threshold are
                mirrored. Set to 0.
            summarypdf: bool
                Option to print a summary PDF.
            verbose: bool
                Print status updates.
        
        Returns:
        --------
            self: Self
                The Riser instance.
        """

        # compute symmetry index if not yet done:
        if self.symmetry_index == []:
            self.compute_symmetry_index()

        if summarypdf:
            if verbose: print(f"Saving summary PDF to {os.getcwd()}", end="...", flush=True)
            p = PdfPages(os.getcwd()+f"\\{self.identifier}_mirrored_profiles.pdf")
        
        typecheck = (type(apply_to_selection) is list)
        for i, name in enumerate(self.name):

            if typecheck is True:
                id_check = apply_to_selection[i]
            else:
                id_check = True

            if id_check is True and self.symmetry_index[i] > symmetry_threshold:

                # calculate the distance of all points to the origin
                dz_points = [(d, z) for d, z in zip(self.d[i], self.z[i])]
                dz_dists = np.array(
                    [np.sqrt(dz[0]**2+dz[1]**2) for dz in dz_points]
                )
                dz_min = np.argmin(dz_dists)
                d_min = self.d[i][dz_min]
                z_min = self.z[i][dz_min]

                # shift all points by d_min, z_min
                self.d[i] -= d_min 
                self.z[i] -= z_min 

                d_pos = self.d[i][self.d[i]>=0]
                z_pos = self.z[i][self.d[i]>=0]

                d_new = np.concatenate((-d_pos[::-1][:-1], d_pos))
                z_new = np.concatenate((-z_pos[::-1][:-1], z_pos)) 

                if summarypdf is True:
                    fg, ax = plt.subplots()
                    ax.set_title(name)
                    ax.plot(d_new, z_new)
                    ax.plot(
                        self.d[i]+d_min, self.z[i]+z_min,
                        linestyle="dashed", linewidth=0.8
                    )
                    ax.axvline(0, linewidth=0.7)
                    ax.axhline(0, linewidth=0.7)
                    fg.savefig(p, format="pdf")
                    plt.close()

                self.d[i] = d_new 
                self.z[i] = z_new
        #print(self.d[2])
        #print(self.z[2])

        if summarypdf: 
            p.close()
            if verbose: print("Done")
        
        return self


    def compute_best_linear_diffusion_fit(
        self,
        init_kt: float = 50.,
        init_a: float = None,
        init_b: float = 0.01,
        init_theta: float = 0.4,
        init_d_off: float = 0.01,
        init_z_off: float = 0.01,
        kt_range: tuple = (0, 10000),
        a_range: tuple = (0, None),
        b_range: tuple = (0, None),
        theta_range: tuple = (0.4, 0.4),
        d_off_range: tuple = (-20, 20),
        z_off_range: tuple = (-10, 10),
        summarypdf: bool = False,
        verbose: bool = True,
        warning_eps: float = -1e-10
    ) -> Self:
        """
        Uses scipy's ``optimize.minimize`` to
        compute a set of parameters leading to a best fit between the analytical
        linear diffusion profile and the measured profile defined by self.d and
        self.z. Initial guesses may be user defined and can lead to
        significantly different results. Initial guesses should not equal zero.

        Parameters:
        -----------
            init_kt: float
                Initial guess for diffusion age.
            init_a:
                Initial guess for the riser height. By default (None) computed
                as ``(self.z[i].max() - self.z[i].min()) / 2``.
            init_b: float
                Initial guess for the far field gradient.
            init_theta: float
                Initial value for the initial gradient at the riser midpoint and
                age zero. This value can only be robustly estimated for some
                risers, especially those that have retained a slope of theta
                around their midpoints. For other risers, this value can vary 
                drastically.
            init_d_off: float
                Initial guess for the offset in along-profile (horizontal)
                direction.
            init_z_off: float
                Initial guess for the offset in elevation (vertical) direction.
            kt_range: tuple
                Bounds of diffusion ange range to be considered by the
                optimization alogrithm.
            a_range: tuple
                Bounds of the (half) riser height to be considered by the
                optimization algorithm.
            b_range: tuple
                Bounds of the far field gradient to be considered by the
                optimization algorithm.
            theta_range: tuple
                Bounds for the initial slope to be considered by the optimization
                algorithm.
            d_off_range: tuple
                Bounds of the offset in along-profile (horizontal) direction to
                be considered by the optimization algorithm.
            z_off_range: tuple
                Bounds of the offset in elevation (vertical) direction to be
                considered by the optimization algorithm.
            summarypdf: bool
                Option to create a summary PDF containing best fits for each
                profile.
            verbose: bool
                Option to print the best fit parameters to the console.

        Returns:
        --------
            self: Self
                The Riser instance.
        """

        #self.best_theta = [init_theta for ind in self.name]  # constant
        params = np.array([1., 1., 1., 1., 1., 1.])  # initial scaled guesses

        # initialize lists for storing results

        opt_sum, b_kt, b_d_off, b_z_off, b_a, b_b, b_theta, b_mf, b_grad = \
            [], [], [], [], [], [], [], [], []


        for i, d in enumerate(self.d):
            
            z = self.z[i]

            if init_a is None:
                init_a1 = (z.max() - z.min()) / 2  # initial guess for a
            else:
                init_a1 = init_a
            
            if a_range == (0, None): 
                a_range1 = (0.5*init_a1, 1.5*init_a1)
            else:
                a_range1 = a_range

            bounds = (kt_range, d_off_range, z_off_range, a_range1, b_range, theta_range)
            scales = np.array([init_kt, init_d_off, init_z_off, init_a1, init_b, init_theta])

            # check if bounds have correct format
            bound_length = [len(b) != 2 for b in bounds]
            if any(bound_length):
                raise Exception("ERROR: Not all bounds are tuples of length two.")

            # check if any initial guesses are zero
            # cannot divide by zero to scale...
            inits = [init_kt, init_d_off, init_z_off, init_a1, init_b, init_theta]
            inits_val = [i == 0 for i in inits]
            if any(inits_val):
                raise Exception("ERROR: Initial parameter guess with zero-value.")


            scaled_bounds = \
                tuple((lb if lb is None else lb / scales[i], ub if ub is None else ub / scales[i])
                      for i, (lb, ub) in enumerate(bounds))

            # sys.exit()
            if verbose is True:
                print(f"Computing best fit for profile {self.name[i]}")

            fit_profile = minimize(
                fun=_compute_misfit_for_optimization,
                x0=params,
                args=(d, z, scales, warning_eps),
                method="Powell",
                bounds=scaled_bounds
            )

            # save summary
            opt_sum.append(fit_profile)
            # save best-fit parameters
            b_kt.append(fit_profile.x[0] * scales[0])
            b_d_off.append(fit_profile.x[1] * scales[1])
            b_z_off.append(fit_profile.x[2] * scales[2])
            b_a.append(fit_profile.x[3] * scales[3])
            b_b.append(fit_profile.x[4] * scales[4])
            b_theta.append(fit_profile.x[5] * scales[5])
            # save best-fit misfit
            b_mf.append(fit_profile.fun)
            # calculate midpoint gradient for best fit:
            b_grad.append(
                analytical_derivative(0, b_kt[-1], b_a[-1], b_b[-1], b_theta[-1])
            )

            if verbose is True:
                print(f"Best fit parameters:"
                      f"\n  kt: {b_kt[-1]:.2f}"
                      f"\n  d_off: {b_d_off[-1]:.3f}"
                      f"\n  z_off: {b_z_off[-1]:.3f}"
                      f"\n  a: {b_a[-1]:.2f}"
                      f"\n  b: {b_b[-1]:.3f}"
                      f"\n  theta: {b_theta[-1]:.2f}"
                      f"\n  RMSE: {b_mf[-1]:.4f}"
                      )

        # save lists to attributes
        self.optimize_summary = opt_sum
        self.best_misfit = b_mf
        self.best_kt = b_kt
        self.best_a = b_a
        self.best_b = b_b
        self.best_theta = b_theta
        self.best_d_off = b_d_off
        self.best_z_off = b_z_off
        self.best_midpoint_grad = b_grad

        if summarypdf is True:

            print(f"\nPrinting summary PDF to {os.getcwd()}", end="... ", flush=True)
            p = PdfPages(os.getcwd() + f"\\{self.identifier}_linear_diffusion_summary.pdf")

            for i, d in enumerate(self.d):
                z_emp = self.z[i]
                d_emp = self.d[i]
                z_ana = analytical_profile(
                    d_emp + self.best_d_off[i],
                    kt=self.best_kt[i],
                    a=self.best_a[i],
                    b=self.best_b[i],
                    theta=self.best_theta[i]
                )
                fg, ax = plt.subplots()
                ax.plot(d_emp, z_emp, linestyle="dotted", c="gray",
                    label="Unadjusted profile")  # original profile
                ax.plot(d_emp + self.best_d_off[i], z_emp + self.best_z_off[i],
                        linestyle="solid", c="gray", label="Shifted profile")
                ax.plot(d_emp + self.best_d_off[i], z_ana, c="red",
                    label="Best fit profile")
                ax.grid(linewidth=0.25, c="darkgray")
                plt.legend(frameon=False)
                ax.set_title(self.name[i])
                ax.set_xlabel("Along-profile distance [m]")
                ax.set_ylabel("Relative elevation [m]")

                plot_text = f"kt = {int(self.best_kt[i])}\n" + \
                            f"a = {self.best_a[i]:.1f}\n" + \
                            f"b = {self.best_b[i]:.3f}\n" + \
                            fr"$\theta = {self.best_theta[i]:.2f}$"+"\n" + \
                            f"\nd_off = {self.best_d_off[i]:.2f}\n" + \
                            f"z_off = {self.best_z_off[i]:.2f}\n" + \
                            f"RMSE = {self.best_misfit[i]:.4f}"
                ax.text(0.75, 0.05, plot_text, horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax.transAxes, fontsize=10)

                fg.savefig(p, format="pdf")
                plt.close()

            p.close()
            print("Done")
        
        return self

    def compute_best_nonlinear_diffusion_fit(
        self,
        t_range: list = [],
        k: float = 1,
        S_c: float = 0.5,
        n: int = 2,
        init_dt: float = 1,
        interp_dx: float = 0.5,
        interp_method = "cubic",
        warning_eps: float = -10e-15,
        save_tz: bool = False,
        summarypdf: bool = False,
        verbose: bool = True,
    ) -> Self:

        """
        compute_best_nonlinear_diffusion_fit: Calculate the best-fit age for
        all profiles contained in the Riser instance. Assumes that ``S_c`` and ``k``
        do not vary between profiles. This method uses the implicit scheme
        based on Perron (2011).

        Parameters:
        -----------
            t_range: list
                List of age boundary tuples. If list is empty,
                upper boundaries are set to self.best_kt[i]*2.
            k: float
                Diffusivity constant in the nonlinear diffusion equation,
                in m^2 / kyr
            S_c: float
                Critical hillslope gradient.
            init_dt: float
                Time step size for the implicit nonlinear diffusion scheme.
            interp_dx: float
                Resolution size for the profile grid used in the implicit
                scheme. Smaller dx will increase computation time, but avoid
                interpolation errors when comparing the modelled profile to the
                measured profile.
            warning_eps: float
                If calculated slope is less than warning_eps, warning is raised.
                Necessary, because negative slopes occur due to floating point
                errors.
            save_tz: bool
                Option to save t and z at every timestep of the numerical modeling.
                May result in significantly larger memory usage.
            interp_method: str or float
                Method used for interpolating the modelled profile to the points
                where elevation was measured. See scipy's interp1d for details.
            summarypdf: bool
                Option to print a summmary pdf containing age estimates for each
                profile.
            verbose: bool
                Option to print process updates to the console while ages are
                calculated.

        Returns:
        --------
            self: Self
                The Riser instance.
        """



        # check if best_a, best_b are defined
        to_check = [self.best_a, self.best_b, self.best_theta]
        checks = [attr == [] for attr in to_check]
        if any(checks):
            raise Exception("a, b or theta not defined. \n"
                "Run compute_best_linear_diffusion_fit before this.")

        # a function that takes as input an age, the z_nl, t_nl matrix etc.
        # computes the nonlinear implicit diffusion and returns the misfit
        # at that age with respect to the measured profile.

        def misfit_nonlin_diff(time, params):

            if time < 0:
                return 9999

            # unpack the parameters
            d_nonlin = params[0]
            z_matrix = params[1]
            z_time = params[2]
            z_profile = params[3] # the measured profile.
            d_profile = params[4] # the distance values of the measured profile
            dx = params[5]
            k = params[6]
            S_c = params[7]
            n = params[8]
            interp_method = params[9]

            # if time is in z_time, compute the misfit right away
            if time in z_time:

                ind = np.where(z_time==time)[0][0]
                interp_fun = interp1d(d_nonlin, z_matrix[ind,:], interp_method)
                z_interp = interp_fun(d_profile)

            else:

                # get next-youngest time step and calculate nonlin diff
                # from there.

                ind = np.where(z_time < time)[0][-1] # last age < time
                new_dt = time - z_time[ind]

                # calculate 1 nonlinear diffusion step using new_dt

                z_at_time, t_at_time = nonlin_diff_perron2011(
                    z_matrix[ind,:],
                    dx,
                    new_dt,
                    1, # only need one time step to reach requested time
                    k,
                    S_c,
                    n,
                    warning_eps=warning_eps
                )

                # interpolate
                interp_fun = interp1d(d_nonlin, z_at_time[-1,:], interp_method)
                z_interp = interp_fun(d_profile)

            # compute misfit for z_interp and z_profile

            # compute weights

            prof_len = d_profile[-1] - d_profile[0]
            f_i = np.zeros(len(z_interp))
            f_i[1:] = (d_profile[1:] - d_profile[:-1]) / prof_len
            f_i *= (len(f_i)/np.sum(f_i))
            # compute misfit

            rmse = \
                np.sqrt(np.sum(f_i*((z_interp-z_profile)**2))/len(z_profile))

            return rmse

        # set upper age boundaries if t_range = []

        if t_range == []:
            t_range = [(0, kt*2) for kt in self.best_kt]

        summaries = []
        best_ts = []
        best_z = []
        best_mf = []
        t_save = [] # save each timestep
        z_save = [] # save profile at each timestep
        d_nonlin_save = [] # x values for nonlinear diffusion
        nonlin_dt = [init_dt]*len(self.name)
        for i, d in enumerate(self.d):

            # first guess for t
            init_t_i = self.best_kt[i]

            if verbose is True:
                print(f"Calculating age for profile {self.name[i]}:")
                print(f"\tUpper t set to {t_range[i][1]:.2f} kyr")

            # estimate needed profile length
            riser_baselength = 2 * self.best_a[i] / np.tan(self.best_theta[i])
            d_length = 5 * riser_baselength

            # check if d_length is longer than measured profile length
            # if not, use the profile length instead.
            if d_length/2 < np.max(np.abs(d)):
                d_length = np.max(np.abs(d))

            d_nonlin = \
                np.arange(-d_length, d_length+interp_dx, interp_dx)
            z_init = analytical_profile(d_nonlin, 0, self.best_a[i],
                self.best_b[i], self.best_theta[i])

            # run the nonlinear diffusion process with the specified dt

            z_nl, t_nl = nonlin_diff_perron2011(
                z_init,
                interp_dx,
                init_dt,
                n_t=int(t_range[i][1]/init_dt)+1,
                k=k,
                S_c=S_c,
                n=n,
                warning_eps=warning_eps
            )

            # build params list for misfit function

            params = tuple([
                d_nonlin,
                z_nl,
                t_nl,
                self.z[i],
                self.d[i],
                interp_dx,
                k,
                S_c,
                n,
                interp_method
            ])
            params = (params,)  # necessary formatting because of the way
            # minizmize passes params into the misfit function

            fit_time = minimize(
                fun=misfit_nonlin_diff,
                x0=np.array([init_t_i]),
                args=params,
                method="Powell",
                bounds=(t_range[i],)
            )

            summaries.append(fit_time)
            best_ts.append(fit_time.x[0])
            best_mf.append(fit_time.fun)
            d_nonlin_save.append(d_nonlin)

            # save the best fit profile

            ind = np.where(t_nl < fit_time.x[0])[0][-1]

            z_onestep, _ = nonlin_diff_perron2011(
                z_nl[ind,:],
                interp_dx,
                fit_time.x[0]-t_nl[ind],
                n_t=1,
                k=k,
                S_c=S_c,
                n=n,
                warning_eps=warning_eps
            )

            if save_tz:
                # save all timesteps (including best fit)
                t_s = t_nl[:ind+1]

                t_save.append(np.append(t_s, fit_time.x[0]))
                # save all profiles (including best fit)
                z_s = z_nl[:ind+1,:]
                z_save.append(np.concatenate((z_s, z_onestep[-1,None]), axis=0))

            #interpolate
            interp_fun = interp1d(d_nonlin, z_onestep[-1,:], interp_method)
            best_z.append(interp_fun(d))

            if verbose is True:
                print(f"\tBest fit age: {best_ts[-1]:.2f} kyr")

        self.nonlin_opt_sum = summaries
        self.nonlin_best_t = best_ts
        self.nonlin_best_z = best_z
        self.nonlin_best_mf = best_mf
        if save_tz:
            self.nonlin_z_matrix = z_save
            self.nonlin_t_times = t_save
        self.nonlin_d = d_nonlin_save
        self.nonlin_dt = nonlin_dt

        if summarypdf is True:
            # pdf of best fit curve and profile for each profile
            if verbose is True:
                print("Creating summary PDF ...")
            p = PdfPages(os.getcwd() + f"\\{self.identifier}_nonlinear_diffusion_summary.pdf")
            for i, name in enumerate(self.name):

                fg, ax = plt.subplots()
                ax.scatter(self.d[i], self.z[i], s=3,
                label="Measured profile")
                ax.plot(self.d[i], self.nonlin_best_z[i], c="red",
                label="Best fit profile")
                ax.set_xlabel("Relative distance [m]")
                ax.set_ylabel("Relative elevation [m]")
                ax.set_title(f"t = {self.nonlin_best_t[i]:.2f} kyr", loc="left")
                ax.set_title(name)
                ax.legend()
                ax.grid(linewidth=0.25, c="darkgray")

                fg.savefig(p, format="pdf")
                plt.close()

            p.close()
            print("Done.")
        
        return self


    def compute_best_nonlinear_diffusion_fit_general(
        self,
        diffusion_law: callable,
        t_range: list = [],
        k: float = 1,
        init_dt: float = 1,
        interp_dx: float = 0.5,
        diff_arg_dict: dict = {},
        interp_method = "cubic",
        summarypdf: bool = False,
        verbose: bool = True,
    ) -> Self:
        """
        Function that takes any differentiable diffusion law, in form
        equal to ``riserfit.nonlin_diff_perron2011`` 
        and performs a fitting routine.
        Any special arguments of the diffusion law (e.g., critical
        slope) are supplied in the diff_arg_dict dictionary.
        
        Parameters:
        -----------
            diffusion_law: callable
                A function like ``riserfit.nonlin_diff_perron2011`` that
                calculates the diffusion of a riser profile and outputs
                a ``np.ndarray`` containing elevation values of every time step
                and a second ``np.ndarray`` containing the cumulative time of the
                time steps.
                The function must at minimum take initial elevations,
                dx, dt, number of time steps, and k as inputs. Any
                additional parameters may be passed by ``diff_arg_dict``.
            t_range: list
                List of upper t boundaries for each profile.
            k: float
                The diffusivity constant of the nonlinear diffusion model.
            init_dt: float
                Size of the time step used for numerical calculations.
            interp_dx: float
                Distance between z grid nodes of the modelled profile.
            diff_arg_dict: dict
                Additional arguments passed on to diffusion_law.
            interp_method: 
                Interpolation method used in scipy.interpolate.interp1d.
            summarypdf: bool
                Option to compile a summary PDF.
            verbose: bool
                Option to print status messages to the console.

        Returns:
        --------
            self: Self
                The Riser instance.
        """

        def nonlin_misfit_general(
            time, diffusion_function, d_nonlin, z_nl, t_nl, z, d, 
            interp_dx, k, interp_method, diff_arg_dict
        ):
            if time < 0: return 9999

            if time in t_nl:

                ind = np.where(t_nl==time)[0][0]
                interp_fun = interp1d(d_nonlin, z_nl[ind,:], interp_method)
                z_interp = interp_fun(d)

            else:

                # get next-youngest time step and calculate nonlin diff
                # from there.

                ind = np.where(t_nl < time)[0][-1] # last age < time
                new_dt = time - t_nl[ind]

                # calculate 1 nonlinear diffusion step using new_dt

                z_at_time, t_at_time = diffusion_function(
                    z_nl[ind,:],
                    interp_dx,
                    new_dt,
                    1, # only need one time step to reach requested time
                    k,
                    **diff_arg_dict
                )

                # interpolate
                interp_fun = interp1d(d_nonlin, z_at_time[-1,:], interp_method)
                z_interp = interp_fun(d)

            # compute misfit for z_interp and z_profile

            # compute weights

            prof_len = d[-1] - d[0]
            f_i = np.zeros(len(z_interp))
            f_i[1:] = (d[1:] - d[:-1]) / prof_len
            f_i *= (len(f_i)/np.sum(f_i))
            # compute misfit

            rmse = \
                np.sqrt(np.sum(f_i*((z_interp-z)**2))/len(z))

            return rmse

        # set upper age boundaries if t_range = []

        if t_range == []:
            t_range = [(0, kt*2) for kt in self.best_kt]

        summaries = []
        best_ts = []
        best_z = []
        best_mf = []
        t_save = [] # save each timestep
        z_save = [] # save profile at each timestep
        d_nonlin_save = [] # x values for nonlinear diffusion
        nonlin_dt = [init_dt]*len(self.name)

        for i, d in enumerate(self.d):

            # first guess for t, just use kt
            init_t_i = self.best_kt[i]

            if verbose is True:
                print(f"Calculating age for profile {self.name[i]}:")
                print(f"\tUpper t set to {t_range[i][1]:.2f} kyr")

            # estimate needed profile length
            riser_baselength = 2 * self.best_a[i] / np.tan(self.best_theta[i])
            d_length = 5 * riser_baselength

            # check if d_length is longer than measured profile length
            # if not, use the profile length instead.
            if d_length/2 < np.max(np.abs(d)):
                d_length = np.max(np.abs(d))

            d_nonlin = \
                np.arange(-d_length, d_length+interp_dx, interp_dx)
            z_init = analytical_profile(d_nonlin, 0, self.best_a[i],
                self.best_b[i], self.best_theta[i])

            # run the nonlinear diffusion process with the specified dt

            z_nl, t_nl = diffusion_law(
                z_init,
                interp_dx,
                init_dt,
                n_t=int(t_range[i][1]/init_dt)+1,
                k=k,
                **diff_arg_dict
            )

            params = (
                diffusion_law,
                d_nonlin,
                z_nl,
                t_nl,
                self.z[i],
                self.d[i],
                interp_dx,
                k,
                interp_method,
                diff_arg_dict
            )

            fit_time = minimize(
                fun=nonlin_misfit_general,
                x0=np.array([init_t_i]),
                args=params,
                method="Powell",
                bounds=(t_range[i],)
            )

            summaries.append(fit_time)
            best_ts.append(fit_time.x[0])
            best_mf.append(fit_time.fun)
            d_nonlin_save.append(d_nonlin)

            # save the best fit profile

            ind = np.where(t_nl < fit_time.x[0])[0][-1]

            z_onestep, _ = diffusion_law(
                z_nl[ind,:],
                interp_dx,
                fit_time.x[0]-t_nl[ind],
                n_t=1,
                k=k,
                **diff_arg_dict
            )

            # save all timesteps (including best fit)
            t_s = t_nl[:ind+1]

            t_save.append(np.append(t_s, fit_time.x[0]))
            # save all profiles (including best fit)
            z_s = z_nl[:ind+1,:]
            z_save.append(np.concatenate((z_s, z_onestep[-1,None]), axis=0))

            #interpolate
            interp_fun = interp1d(d_nonlin, z_onestep[-1,:], interp_method)
            best_z.append(interp_fun(d))

            if verbose is True:
                print(f"\tBest fit age: {best_ts[-1]:.2f} kyr")

        self.nonlin_opt_sum = summaries
        self.nonlin_best_t = best_ts
        self.nonlin_best_z = best_z
        self.nonlin_best_mf = best_mf
        self.nonlin_z_matrix = z_save
        self.nonlin_t_times = t_save
        self.nonlin_d = d_nonlin_save
        self.nonlin_dt = nonlin_dt

        if summarypdf is True:
            # pdf of best fit curve and profile for each profile
            if verbose is True:
                print("Creating summary PDF ...")
            p = PdfPages(os.getcwd() + f"\\{self.identifier}_nonlinear_diffusion_summary.pdf")
            for i, name in enumerate(self.name):

                fg, ax = plt.subplots()
                ax.scatter(self.d[i], self.z[i], s=3,
                label="Measured profile")
                ax.plot(self.d[i], self.nonlin_best_z[i], c="red",
                label="Best fit profile")
                ax.set_xlabel("Relative distance [m]")
                ax.set_ylabel("Relative elevation [m]")
                ax.set_title(f"t = {self.nonlin_best_t[i]:.2f} kyr", loc="left")
                ax.set_title(name)
                ax.legend()
                ax.grid(linewidth=0.25, c="darkgray")

                fg.savefig(p, format="pdf")
                plt.close()

            p.close()
            print("Done.")
        
        return self

    def add_cn_terrace_ages(
        self,
        filepath: str,
        terrace_col: str = "terrace",
        age_col: str = "age",
        sigma_col: str = "sigma",
        pd_dict: dict = {}
    ) -> Self:
        """
        Assiociates the riserfit.Riser.terrace parameter of a
        class instance with terrace ages specified in a ``.csv``
        file.

        Parameters:
        -----------
            filepath: str
                Path to subdirectory containing .csv file.
            terrace_col: str
                Column name with terrace ids/names.
            age_col: str
                Column name with terrace ages.
            sigma_col: str
                Column name with terrace age standard deviations.
            pd_dict: dict
                Any arguments passed to ``pandas.read_csv``.

        Returns:
        --------
            self: Self
                The Riser instance.
        """

        df = pd.read_csv(f"{os.getcwd()}\\{filepath}", **pd_dict)
        df_terrace = df[terrace_col].tolist()
        df_ages = df[age_col].tolist()
        df_sigma = df[sigma_col].tolist()

        profile_age = []
        profile_sigma = []
        # for each profile, find df_terrace...
        for i, name in enumerate(self.name):
            t = self.terrace[i]
            t_id = np.where(np.array(df_terrace)==t)[0]
            if len(t_id) == 0:
                profile_age.append(np.nan)
                profile_sigma.append(np.nan)
            else:
                profile_age.append(df_ages[t_id[0]])
                profile_sigma.append(df_sigma[t_id[0]])

        self.terrace_age = profile_age
        self.terrace_age_sigma = profile_sigma
        
        return self

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: list
    ) -> Self:
        """
        Generic function to add a new parameter to the class instance.
        
        Parameters:
        -----------
            parameter_name: str
                The name of the new parameter.
            parameter_values: list
                The values (a list) of the new parameter.
                
        Returns:
        --------
            self: Self
                The Riser instance.
        """
        
        setattr(self, parameter_name, parameter_values)
        return self 
    
    
    def add_parameter_from_file(
        self, 
        parameter_name,
        filepath,
        name_col: str,
        parameter_col: str = None,
        pd_dict: dict = {},
        match_type: str = "exact"
    ) -> Self:
        """
        Add a new parameter from a csv file to the riser instance.
        
        Parameters:
        -----------
            parameter_name: str
                Name of the new parameter to be added. available
                as ``self.parameter_name``.
            filepath: str
                Sub-path to CSV file.
            name_col: str
                Column name containing the names as they appear in 
                self.name
            parameter_col: str
                The column containing the parameter values to be added.
                If None, assumed to be the same as parameter_name.
            pd_dict: dict
                Passed to ``pandas.read_csv``.
            match_type: str
                Determines how the names in name_column are compared to those
                in self.name. Default is "exact", meaning that the names have
                to match exactly. Alternative is "partial", meaning that the
                file names have to be contained in the riser.name names. I.e.,
                a file name of "aaa" will be matched with "aaa_0" (but not
                vice versa). If multiple occurences are found using either type
                of matching, the first occurence is used.
            
        Returns:
        --------
            self: Self
                The Riser instance.
        """
        
        df = pd.read_csv(os.getcwd()+f"\\{filepath}", **pd_dict)
        if name_col is None: name_col = parameter_name
        
        parameter_out = [np.nan]*len(self.name)
        for i, name in enumerate(self.name):
            # find row where name entry in file is name
            if match_type == "exact":
                id = [
                    True if (name==df_name) else False
                    for df_name in df[name_col]
                ]
            elif match_type == "partial":
                id = [
                    True if (df_name in name) else False
                    for df_name in df[name_col]
                ]
            else:
                raise Exception("match_type not in ['exact', 'partial']")
            # use first entry, if there is any True entry...
            if any(id):
                parameter_out[i] = df.loc[id][parameter_col].iloc[0]

        setattr(self, parameter_name, parameter_out)
        
        return self
        
    def add_cn_ages(
        self,
        filepath: str,
        name_col: str = "Profile",
        age_col: str = "avgCN",
        sigma_col: str = "sigma",
        reliability = None,
        reliability_dict = None,
        verbose: bool = False,
        pd_dict: dict = {}
    ) -> Self:
        """
        add_cn_ages: Function to associate riser profiles with available
        cosmogenic nuclide (CN) ages. CN ages are loaded from a .csv
        file defined by the filepath and infile.
        This file should consist of at least:
        #. A column containing the profile name. If the name in the
        file is profileA, but profileA has been split into
        profileA_0, profileA_1, etc., it will still be found and the
        CN age associated with profileA will be assigned to
        profileA_0, profileA_1, etc.
        #. A column containing the sample (or mean sample) age.
        #. A column containing CN uncertainty (1sigma or similar).
        #. (Optional) A column indicating the reliability of CN
        ages.

        Parameters:
        -----------
            filepath: str
                Path to the folder containing a ``.csv`` file with CN ages.
            infile: str
                Name of the ``.csv`` file containing the CN ages.
            name_col: str
                Name of the column containing the profile names.
            age_col: str
                Name of the column containing the CN ages.
            sigma_col: str
                Name of the column containing the CN age uncertainties.
            reliability: str
                If some CN ages are to be treated with caution, indicate
                the column name storing that information here.
            reliability_dict: dict
                Dictionary containing reliability information and color
                for plotting, e.g.: {"y": "blue", "n": "gray"}.
            pd_dict: dict
                Passed on to ``pandas.read_csv()``.

        Returns:
        --------
            self: Self
                The Riser instance.
        """

        df = pd.read_csv(os.getcwd()+f"\\{filepath}", **pd_dict)

        cn_ages = [np.nan for i in range(0, len(self.name))]
        cn_age_sigma = [0 for i in range(0, len(self.name))]
        cn_reliability = ["" for i in range(0, len(self.name))]

        if verbose: print("\nSearching for CN ages ...")

        for i, cn_name in enumerate(df[name_col]):

            riser_index = [ind for ind, name in enumerate(self.name) if cn_name in name]
            #print(f" Ind list:{riser_index} \n CN name: {cn_name} \n Names: {self.name}")
            if len(riser_index) > 0:

                for ind in riser_index:
                    cn_ages[ind] = df.iloc[i][age_col]
                    cn_age_sigma[ind] = df.iloc[i][sigma_col]
                    if reliability != None:
                        cn_reliability[ind] = df.iloc[i][reliability]

                    if verbose: print(f"Found age for Riser class profile {self.name[ind]}")

            else:

                if verbose: print(f"Could not match {cn_name} from file with any Riser class profile.")

        self.cn_age = cn_ages
        self.cn_age_sigma = cn_age_sigma
        self.cn_age_reliability = cn_reliability
        
        return self
    
    def add_terrace_generation(
        self,
        filepath: str,
        name_col: str,
        terrace_col: str,
        verbose: bool = False,
        pd_dict: dict = {}
    ) -> Self:
        """
        Add terrace generations (names or numbers) to each riser profile.

        Parameters:
        -----------
            filepath: str
                Subdirectory and filename containing a ``.csv`` file with
                profile names and terrace generations.
            name_col: str
                Column name in ``.csv`` file containing profile names.
            terrace_col: str
                Column name in ``.csv`` file containing the terrace generation.
            verbose: str
                Option to print status updates.
            pd_dict: dict
                Any options passed on to ``pandas.read_csv``.

        Returns:
        --------
            self: Self
                The Riser instance.
        """

        df = pd.read_csv(f"{os.getcwd()}\\{filepath}", **pd_dict)
        df_names = df[name_col].tolist()
        df_terraces = df[terrace_col].tolist()
        terrace_info = np.full(len(self.name), np.nan, dtype="object")

        for i, dfn in enumerate(df_names):
            # check where dfn appears in self.name
            ids = [True if dfn in rn else False for rn in self.name]
            terrace_info[ids] = df_terraces[i]
            if verbose: print(f"Found match for profile {dfn} from file")
        
        self.terrace = list(terrace_info)
        
        return self

    def calculate_kt_uncertainty(
        self,
        dt: float = 5,
        max_iteration: int = 1000,
        summarypdf: bool = False,
        ascending: bool = False,
        verbose: bool = True
    ) -> Self:

        """
        Calculates uncertainties in linear kt. Analogous to 
        `Wei et al. 2015 <https://doi.org/10.1016/j.jseaes.2015.02.016>`_.

        Parameters:
        -----------
            dt: float
                Step size when searching for inversion bounds on kt.
            max_iteration: int
                Maximum number of iterations when searching for 
                the inversion bounds in kt.
            summarypdf: bool
                Option to plot all diffusion ages against profile name,
                including uncertainty.
            ascending: bool
                Option to plot ages in ascending order.
            verbose: bool
                Option to print results to console output.

        Returns:
        --------
            self: Self
                The Riser instance.
        """
 
        l_kt,u_kt = [], []

        for i, _ in enumerate(self.name):
            
            geom_params = {
                "kt": self.best_kt[i],
                "a": self.best_a[i],
                "b": self.best_b[i],
                "theta": self.best_theta[i]
            }
            sigma = calculate_wei_2015_sigma(
                self.z[i]-analytical_profile(
                    self.d[i], self.best_kt[i], self.best_a[i],
                    self.best_b[i], self.best_theta[i]
                )
            )
             
            lb, ub = _lin_invert_uncertainty(
                d=self.d[i],
                z=self.z[i],
                kt_best=self.best_kt[i],
                sigma=sigma,
                min_mse=self.best_misfit[i]**2,
                dt=dt,
                max_iteration=max_iteration,
                geom_params=geom_params
            )
            
            if verbose == True:
                print(f"kt bounds for {self.name[i]}:")
                print(f"\t Lower kt = {lb:.2f}")
                print(f"\t Best kt = {self.best_kt[i]:.2f}")
                print(f"\t Upper kt = {ub:.2f}")
                
            l_kt.append(lb)
            u_kt.append(ub)

        self.lower_kt = l_kt
        self.upper_kt = u_kt

        if summarypdf is True:
            # plot all profile ages on y, names on x
            y_upper = np.array(self.upper_kt)-np.array(self.best_kt)
            y_lower = np.array(self.best_kt)-np.array(self.lower_kt)

            if ascending is True:
                order_ID = np.argsort(self.best_kt)
                y_uncert = np.array([y_lower[order_ID], y_upper[order_ID]])
                kt_age = np.array(self.best_kt)[order_ID]
                prof_name = np.array(self.name)[order_ID]
            else:
                y_uncert = np.array([y_lower, y_upper])
                kt_age = np.array(self.best_kt)
                prof_name = np.array(self.name)

            plot_at = [i for i in range(0,len(self.best_kt))]
            fg, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.1)
            ax.errorbar(plot_at, kt_age, yerr=y_uncert, fmt="o",
            ecolor="gray", capsize=3, elinewidth=0.5,
            capthick=0.5, c="blue")
            ax.set_xticks(plot_at)
            ax.set_xticklabels(prof_name)
            ax.tick_params("x", labelrotation=90)
            ax.set_ylabel(r"$kt$ [m$^2$]")
            fg.tight_layout()
            plt.savefig(os.getcwd()+f"\\{self.identifier}_summary_kt_uncertainty.pdf",
                bbox_inches="tight")
        
        return self

    def calculate_nonlin_t_uncertainty(
        self,
        k: float = 1,
        S_c: float = 0.5,
        n: float = 2,
        dt: float = 1., 
        warning_eps: float = -10e-15,
        max_iteration: int = 10_000,
        summarypdf: bool = False,
        verbose: bool = True,
    ) -> Self:
        """
        Calculate uncertainty bounds for
        nonlinear age. Analogous to 
        `Wei et al. 2015 <https://doi.org/10.1016/j.jseaes.2015.02.016>`_.

        Parameters:
        -----------
            k: float
                Nonlinear diffusion coefficient
            S_c: float
                Critical slope.
            n: int
                Exponent in the nonlinear diffusion equation.
            dt: float
                Time step for diffusion model.
            warning_eps: float
                Raise warning if slopes of the modelled profiles
                are smaller than warning_eps.
            max_iteration: int
                How far ahead t are computed when searching for the upper
                uncertainty bound. The upper limit is determined by 
                dt*max_iteration
            summarypdf: bool
                Option to plot all diffusion ages against profile name,
                including uncertainty.
            verbose: bool
                Option to print results to console output.

        Returns:
        --------
            self: Self
                The Riser instance.
        """
        
        l_t = []
        u_t = []
        
        for i, _ in enumerate(self.name):
            
            # calculate wei et al sigma
            sigma = calculate_wei_2015_sigma(
                self.z[i] - self.nonlin_best_z[i]
            )
            geom_params = {
                "kt": 0,
                "a": self.best_a[i],
                "b": self.best_b[i],
                "theta": self.best_theta[i]
            }
            
            lb, ub = _nonlin_invert_uncertainty(
                d=self.d[i],
                z=self.z[i],
                nonlin_d=self.nonlin_d[i],
                z_best=self.nonlin_best_z[i],
                t_best=self.nonlin_best_t[i],
                dt=dt,
                sigma=sigma,
                min_mse=self.nonlin_best_mf[i]**2,
                S_c=S_c,
                geom_params=geom_params,
                max_iteration=max_iteration,
                k=k,
                n=n,
                warning_eps=warning_eps
            )

        
            l_t.append(lb)
            u_t.append(ub)

            if verbose == True:
                print(f"Calculating t bounds for {self.name[i]}:")
                print(f"\t Lower t = {lb:.2f} kyr")
                print(f"\t Best t = {self.nonlin_best_t[i]:.2f} kyr")
                print(f"\t Upper t = {ub:.2f} kyr")

        self.nonlin_lower_t = l_t
        self.nonlin_upper_t = u_t

        if summarypdf is True:
            # plot all profile ages on y, names on x
            y_upper = np.array(self.nonlin_upper_t)-np.array(self.nonlin_best_t)
            y_lower = np.array(self.nonlin_best_t)-np.array(self.nonlin_lower_t)

            y_uncert = np.array([y_lower, y_upper])
            kt_age = np.array(self.nonlin_best_t)
            prof_name = np.array(self.name)

            plot_at = [i for i in range(0,len(self.nonlin_best_t))]
            fg, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.1)
            ax.errorbar(plot_at, kt_age, yerr=y_uncert, fmt="o",
            ecolor="gray", capsize=3, elinewidth=0.5,
            capthick=0.5, c="blue")
            ax.set_xticks(plot_at)
            ax.set_xticklabels(prof_name)
            ax.tick_params("x", labelrotation=90)
            ax.set_ylabel(r"t [kyr]")
            fg.tight_layout()
            plt.savefig(os.getcwd()+f"\\{self.identifier}_summary_nonlin_uncertainty.pdf",
                bbox_inches="tight")
        
        return self

    def jackknife_linear_diffusion_fit(
        self,
        iterations: int = 10,
        fraction_reduction: float = 0.1,
        every_kth: int = 0,
        verbose: bool = True,
        summarypdf: bool = False,
        warning_eps: float = -1e-10,
        lin_diff_dict: dict = {}
    ) -> Self:
        """
        Calculates a range of kt and other parameters for linear diffusion
        based on a profile consisting only of a subset of points from
        the original profile.

        Parameters:
        -----------
            iterations: int
                The number of resamples undertaken for each profile.
            fraction_reduction: float
                Fraction of points to be left out of the subset.
            every_kth: int
                If larger than 0, used for sampling points. 
            verbose: bool
                Option to print progress statements to console.
            summarypdf: bool
                Print a PDF showing jackknifing histograms.
            lin_diff_dict: dict
                Additional arguments passed to 
                ``Riser.compute_best_linear_diffusion_fit()``.

        Returns:
        --------
            self: Self
                The Riser instance.

        """

        rng = np.random.default_rng()

        # save original d, z
        # save original linear diffusion fit results
        save_z = self.z.copy()
        save_d = self.d.copy()
        save_a, save_b = self.best_a.copy(), self.best_b.copy()
        save_doff, save_zoff = self.best_d_off.copy(), self.best_z_off.copy()
        save_kt, save_sum, = self.best_kt.copy(), self.optimize_summary.copy()
        save_mf, save_midgrad = self.best_misfit.copy(), self.best_midpoint_grad.copy()

        # save results of each iteration here
        # this is terrible but there were some problems with
        # identical arrays in different variables...
        #prof_z = [[] for _ in range(0, len(self.name))]
        #prof_d = [[] for _ in range(0, len(self.name))]
        prof_a = [[] for _ in range(0, len(self.name))]
        prof_b = [[] for _ in range(0, len(self.name))]
        prof_doff = [[] for _ in range(0, len(self.name))]
        prof_zoff = [[] for _ in range(0, len(self.name))]
        prof_kt = [[] for _ in range(0, len(self.name))]
        prof_sum = [[] for _ in range(0, len(self.name))]
        prof_mf = [[] for _ in range(0, len(self.name))]
        prof_midgrad = [[] for _ in range(0, len(self.name))]

        if verbose: print("\nLinear jackknife algorithm:")

        # only need one iteration to check every_kth
        if every_kth > 0: iterations = 1

        for i in range(0, iterations):

            #if verbose: sys.stdout.write('\033[2K\033[1G')
            if verbose: sys.stdout.write("\r"+f"Starting iteration {i+1}/{iterations}")

            # subsample each profile
            for j, name in enumerate(self.name):

                sample_size = int(len(save_d[j])*(1-fraction_reduction))
                indices = np.array(list(range(0, len(save_d[j])-1)))
                
                if every_kth > 0:
                    id_sample = indices[::every_kth]
                else:
                    id_sample = \
                        np.sort(rng.choice(a=indices, size=sample_size, replace=False))

                self.d[j] = save_d[j][id_sample]
                self.z[j] = save_z[j][id_sample]
                #prof_z[j].append(self.z[j])
                #prof_d[j].append(self.d[j])

            # are all of the indices just the same?

            # compute linear diffusion fit for all profiles

            self.compute_best_linear_diffusion_fit(
                verbose=False,
                warning_eps = warning_eps,
                **lin_diff_dict
            )

            # append results...
            for j, name in enumerate(self.name):
                prof_kt[j].append(self.best_kt[j])
                prof_a[j].append(self.best_a[j])
                prof_b[j].append(self.best_b[j])
                prof_doff[j].append(self.best_d_off[j])
                prof_zoff[j].append(self.best_z_off[j])
                prof_sum[j].append(self.optimize_summary[j])
                prof_mf[j].append(self.best_misfit[j])
                prof_midgrad[j].append(self.best_midpoint_grad[j])

        if verbose: print("... Done")
        # save to self.jack_ ...

        #self.jack_d = prof_d
        #self.jack_z = prof_z
        self.jack_kt = prof_kt
        self.jack_a = prof_a
        self.jack_b = prof_b
        self.jack_d_off = prof_doff
        self.jack_z_off = prof_zoff
        self.jack_optimize_summary = prof_sum
        self.jack_misfit = prof_mf
        self.jack_midpoint_gradient = prof_midgrad

        # replace original d, z etc.

        self.d = save_d 
        self.z = save_z
        self.best_a = save_a
        self.best_b = save_b
        self.best_kt = save_kt
        self.best_misfit = save_mf
        self.optimize_summary = save_sum
        self.best_d_off = save_doff
        self.best_z_off = save_zoff
        self.best_midpoint_grad = save_midgrad

        if summarypdf:
            #cGray = [0.2, 0.2, 0.2]
            if verbose: 
                print(f"Saving summary PDF to {os.getcwd()}", 
                    end="... ", flush=True
                )
            p = PdfPages(os.getcwd() + f"\\{self.identifier}_linear_jackknife.pdf")
            for i, name in enumerate(self.name):

                # create text:
                t1 = f"std = {np.std(self.jack_kt[i]):.2f}\n"
                #t2 = f"upper kt = {self.upper_kt[i]:.2f}\n"
                #t3 = f"lower kt = {self.lower_kt[i]:.2f}"
                fg, ax = plt.subplots()
                ax.hist(self.jack_kt[i], color="lightgray", density=True)
                ax.axvline(np.mean(self.jack_kt[i]), c="red",
                    linestyle="dashed", linewidth=0.8, label="mean")
                if len(self.best_kt) > 0:
                    ax.axvline(self.best_kt[i], c="blue",
                        linestyle="dashed", linewidth=0.8, label="best")
                ax.set_xlabel(r"kt [m$^2$]")
                ax.set_ylabel(r"Density")
                ax.set_title(name)
                ax.legend(loc="upper right")
                ax.text(0.1, 0.8, t1, transform=ax.transAxes)
                #ax.text
                fg.savefig(p, format="pdf")
                plt.close()

            p.close()
            if verbose: print("Done")
        
        return self

    def jackknife_nonlinear_diffusion_fit(
        self,
        iterations: int = 10,
        fraction_reduction: float = 0.1,
        verbose: bool = True,
        summarypdf: bool = False,
        nonlin_diff_dict: dict = {}
    ) -> Self:
        
        """
        Calculates a range of kt and other parameters for nonlinear diffusion
        based on a profile consisting only of a subset of points from
        the original profile.

        Parameters:
        -----------
            iterations: int
                The number of resamples undertaken for each profile.
            fraction_reduction: float
                Fraction of points to be removed  from the set of points.
            verbose: bool
                Option to print progress statements to console.
            summarypdf: bool
                Print a PDF showing jackknifing histograms.
            nonlin_diff_dict: dict
                Additional arguments passed to 
                ``Riser.compute_best_nonlinear_diffusion_fit()``.

        Returns:
        --------
            self: Self
                The Riser instance.

        """

        rng = np.random.default_rng()

        # save original d, z
        # save original linear diffusion fit results
        save_z = self.z.copy()
        save_d = self.d.copy()
        
        # save results of compute_best_nonlinear_diffusion_fit()
        save_nbt = self.nonlin_best_t.copy()
        save_nos = self.nonlin_opt_sum.copy()
        save_nbz = self.nonlin_best_z.copy()
        save_nzm = self.nonlin_z_matrix.copy()
        save_ntt = self.nonlin_t_times.copy()
        save_nd = self.nonlin_d.copy()
        save_nbm = self.nonlin_best_mf.copy()
        save_nd = self.nonlin_dt.copy()

        # save results of each iteration here
        # this is terrible but there were some problems with
        # identical arrays in different variables...
        prof_nonlin_best_t = [[] for _ in self.name]

        if verbose: print("\nNonlinear jackknife algorithm:")
        for i in range(0, iterations):

            if verbose: sys.stdout.write("\r"+f"Starting iteration {i+1}/{iterations}")

            # subsample each profile
            for j, name in enumerate(self.name):

                sample_size = int(len(self.x[j])*(1-fraction_reduction))
                indices = np.array(list(range(0, len(self.x[j]))))
                id_sample = np.sort(rng.choice(a=indices, size=sample_size, replace=False))

                self.d[j] = save_d[j][id_sample]
                self.z[j] = save_z[j][id_sample]

            # compute nonlinear diffusion fit for all profiles

            self.compute_best_nonlinear_diffusion_fit(
                verbose=False,
                **nonlin_diff_dict
            )

            # append results...
            for j, name in enumerate(self.name):
                prof_nonlin_best_t[j].append(self.nonlin_best_t[j])

        # save to self.jack_ ...

        #self.jack_d = prof_d
        #self.jack_z = prof_z
        self.nonlin_jack_best_t = prof_nonlin_best_t

        # replace original d, z etc.
        self.d = save_d 
        self.z = save_z
        self.nonlin_best_t = save_nbt
        self.nonlin_opt_sum = save_nos
        self.nonlin_best_z = save_nbz
        self.nonlin_z_matrix = save_nzm
        self.nonlin_t_times = save_ntt
        self.nonlin_d = save_nd
        self.nonlin_best_mf = save_nbm
        self.nonlin_dt = save_nd

        if summarypdf:
            #cGray = [0.2, 0.2, 0.2]
            if verbose: print("Creating summary PDF", end="... ", flush=True)
            p = PdfPages(os.getcwd() + f"\\{self.identifier}_nonlinear_jackknife.pdf")
            for i, name in enumerate(self.name):

                # create text:
                t1 = f"std = {np.std(self.nonlin_jack_best_t[i]):.2f}\n"
                if self.nonlin_upper_t != []:
                    t2 = f"upper kt = {self.nonlin_upper_t[i]:.2f}\n"
                    t3 = f"lower kt = {self.nonlin_lower_t[i]:.2f}"
                else:
                    t2 = ""
                    t3 = ""
                fg, ax = plt.subplots()
                ax.hist(self.nonlin_jack_best_t[i], color="lightgray", density=True)
                ax.axvline(np.mean(self.nonlin_jack_best_t[i]), c="red",
                    linestyle="dashed", linewidth=0.8, label="mean")
                if len(self.nonlin_best_t) > 0:
                    ax.axvline(self.nonlin_best_t[i], c="blue",
                        linestyle="dashed", linewidth=0.8, label="best")
                ax.set_xlabel(r"kt$_{nl}$ [m$^2$]")
                ax.set_ylabel(r"Density")
                ax.set_title(name)
                ax.legend(loc="upper right")
                ax.text(0.1, 0.8, t1+t2+t3, transform=ax.transAxes)
                #ax.text
                fg.savefig(p, format="pdf")
                plt.close()

            p.close()
            if verbose: print("Done")
        
        return self

    def calculate_parameter_uncertainties(
        self,
        epsilon: Union[ArrayLike, float] = 1.4901161193847656e-08
    ) -> Self:
        """
        Calculates confidence bands by treating the parameter estimation
        as a nonlinear regression problem 
        (`see here <https://www.youtube.com/watch?v=3IgIToOV2Wk>`_).
        Results are stored in self.parameter_uncertainties.
        
        Parameters:
        -----------
            epsilon: Union[ArrayLike, float]
                The epsilon (delta value) used for calculating
                derivatives numerically.
                
        Returns:
        --------
            self: Self
                The Riser instance.
        """

        uncerts = []
        for i, name in enumerate(self.name):

            x_vals = self.d[i]
            param_vals = {
                "kt": self.best_kt[i], 
                "a": self.best_a[i],
                "b": self.best_b[i],
                "theta": self.best_theta[i]
            }

            jacobian = calculate_function_jacobian(
                analytical_profile,
                x_values=x_vals,
                parameter_values=param_vals,
                epsilon=epsilon
            )

            z_best = analytical_profile(
                self.d[i], self.best_kt[i], self.best_a[i],
                self.best_b[i], self.best_theta[i]
            )

            # weights
            prof_len = self.d[i][-1] - self.d[i][0]
            f_i = np.zeros(len(self.d[i]))
            f_i[1:] = (self.d[i][1:] - self.d[i][:-1]) / prof_len
            f_i *= (len(f_i)/np.sum(f_i))

            sos = np.sum(1*(z_best-self.z[i])**2)
            
            cov_matrix = riser_covariance_matrix(
                jacobian=jacobian,
                sum_of_squares=sos
            )

            uncerts.append(cov_matrix)

        self.parameter_uncertainties = uncerts
        
        return self
    
    def construct_StatsMC_instance(
        self,
        kt_parameter_name: str = "best_kt",
        kt_lb_parameter_name: str = "lower_kt",
        kt_ub_parameter_name: str = "upper_kt",
        t_parameter_name: str = "cn_age",
        t_sigma_parameter_name: str = "cn_age_sigma",
        identifier: str = None
    ) -> Tuple:
        """
        This function return a tuple of ``StatsMC`` instances. How many instances
        are created depends on the number of different entries in the
        ``t_parameter_name`` list. I.e., if all risers have the same age,
        only one instance is returned.
        If any risers have the entry np.nan associated, they are grouped into
        their own ``StatsMC`` entry and take up the last position in the returned
        tuple.
        
        Parameters:
        -----------
            kt_parameter_name: str
                Determines whether to use linear or nonlinear diffusion, so
                should be either "best_kt" (for linear diffusion) or 
                "nonlin_best_t" (for nonlinear diffusion). Default is linear
                diffusion.
            kt_lb_parameter_name: str
                Determines if bounds from linear or nonlinear diffusion are 
                to be used. Default is "lower_kt" (linear diffusion)l
                alternative is "nonlin_lower_t" (nonlinear diffusion).
            kt_ub_parameter_name: str
                Determines if bounds from linear or nonlinear diffusion are 
                to be used. Default is "upper_kt" (linear diffusion)
                alternative is "nonlin_upper_t" (nonlinear diffusion).
            t_parameter_name: str
                Parameter storing age information. This can also be an array
                of length ``n`` for each riser, indicating that there are multiple
                available ages to be used. If so, the t PDF used for MC 
                simulations will be a composite Gaussian distribution. Default
                is "cn_age".
                The data stored in ``getattr(self, t_parameter_name)[i]`` needs to
                be either a numerical value (float, ...) or a ``np.array``. Lists
                are not supported.
            t_sigma_parameter_name: str
                Parameter storing the uncertainty information associated with
                each entry in ``t_parameter_name``. These entries must have the same
                format as those in ``t_parameter_name``. I.e., if the latter is 
                an array of length n for a single riser, the corresponding 
                information in ``t_sigma_parameter_name`` also needs to be an 
                array of length ``n``.
            identifier: str
                Set the ``StatsMC.identifier`` property of the created instance.
                If ``None``, ``Riser.indentifier`` is used.
                
        Returns:
        --------
            statsMC: Tuple
                A length ``n`` tuple containing instances of ``riserfit.StatsMC``.
                If some risers do not have an age assigned, they are grouped
                in the last returned ``StatsMC`` instance.
        """
        
        def _selector(par, ids):
            return [x for i, x in enumerate(par) if i in ids]
        
        ts = getattr(self, t_parameter_name)
        ts_sigma = getattr(self, t_sigma_parameter_name)
        
        kts = getattr(self, kt_parameter_name)
        kts_lb = getattr(self, kt_lb_parameter_name)
        kts_ub = getattr(self, kt_ub_parameter_name)
        
        # find unique entries in ts:
        unique_t = [ts[0]]
        for t in ts:
            if not any([np.array_equal(t, u_t, equal_nan=True) for u_t in unique_t]):
                unique_t.append(np.array(t))
        instance_list = []
        
        for u_t in unique_t:
            
            # nan ages are grouped into their own instance.
            # use list comprehension to avoid accidental concats by numpy...
            ids = [
                i for i, t in enumerate(ts)
                if np.array_equal(u_t, t, equal_nan=True)
            ]

            kts_id = _selector(kts, ids)
            lb_id = _selector(kts_lb, ids)
            ub_id = _selector(kts_ub, ids)
            t_id = _selector(ts, ids)[0]
            ts_id = _selector(ts_sigma, ids)[0]
            
            if identifier != None:
                set_id = identifier 
            else: 
                set_id = self.identifier
                
            instance_list.append(
                StatsMC(
                    kt=kts_id,
                    lb_kt=lb_id,
                    ub_kt=ub_id,
                    t=t_id,
                    t_sigma=ts_id,
                    identifier=set_id
                )
            )

        return tuple(instance_list)
    
    """
    Some functions that are useful for contextual analysis and plotting
    """
    
    def extract_profile_elevation(
        self,
        rasterpath,
        band_no: int = 1,
        surface: str = "lower",
        method: str = "linear",
        **rio_open_params
    ) -> Self:

        """
        Extract profile elevations using a DEM or REM. Depending on the value
        of surface, the first or last point of each profile is used to calculate
        the profile elevation.

        Parameters:
        -----------
            rasterpath: str
                Path to file containing a raster with river-relative elevations
            band_no: int
                Band containing elevation.
            surface: str
                Extract elevation from the 'upper' or 'lower' terrace surface.
            method: str
                Interpolation method used to extract relative elevations. Can
                take any value allowed by ``scipy.interpolate.interpn()``.

        Returns:
        --------
            self: Self
                The riser instance.

        """

        # construct points list:
        if surface == "lower":
            # approximate using first point in each profile
            x = [xx[0] for xx in self.x]
            y = [yy[0] for yy in self.y]
        elif surface == "upper":
            # approximate using last point in each profile (upper surface)
            x = [xx[-1] for xx in self.x]
            y = [yy[-1] for yy in self.y]
        else:
            raise Exception("surface must be either 'lower' or 'upper'")
        points = [(y, x) for x, y in zip(x, y)]
        elevations = sample_raster_at_points(
            rasterpath,
            points,
            method,
            band_no,
            **rio_open_params
        )

        self.profile_elevation = list(elevations)
        
        return self

    def calculate_upstream_distance(
        self,
        riverpath,
        surface: str = "lower",
        df_x: str = "x",
        df_y: str = "y",
        df_dist: str = "distance",
        invert_distances: bool = False
    ) -> Self:
        """
        Takes a ``.csv`` file containing points describing a river path and calculates
        the along-stream distance for each Riser profile (based on the first or last
        point of each profile).

        Parameters:
        -----------
            riverpath: str
                Path to ``.csv`` file containing ``(x, y)`` points describing the
                river path.
            df_x: str
                Name of the ``.csv`` column containing the ``x`` coordinate.
            df_y: str
                Name of the ``.csv`` column containing the ``y`` coordinate.
            df_dist: str
                Name of the column containing the distances between points along,
                the river line, if already calculated. Set to ``""`` if 
                not present in ``.csv`` file.
            invert_distances: bool
                Option to flip the points. Will calculate distances in the
                opposite way.

        Returns:
        --------
            self: Self
                The Riser instance.

        """
        # read river .csv file
        river_df = pd.read_csv(os.getcwd()+"\\"+riverpath)

        # get x, y coordinates:
        river_x = river_df[df_x].to_numpy()
        river_y = river_df[df_y].to_numpy()

        # change order if wanted.
        if invert_distances is True:
            river_x = river_x[::-1]
            river_y = river_y[::-1]

        # if df_dist == "", calculate along-stream distances for each node
        river_diff_dists = np.zeros(river_x.shape)
        if df_dist == "":
            river_diff_dists[1:] = np.sqrt((river_x[1:] - river_x[:-1])**2
                + (river_y[1:] - river_y[:-1])**2)
            river_dists = np.cumsum(river_diff_dists)
            df_dist = "distance"
            river_df["distance"] = river_dists
        else:
            river_dists = river_df[df_dist].to_numpy()

            # only change order if values are taken from DataFrame,
            # else calculated from already rearranged coordinate points.
            if invert_distances is True:
                river_dists_new = np.zeros(river_dists.shape)
                river_dists_new[1:] = \
                    river_dists[1:][::-1] - river_dists[:-1][::-1]
                river_dists = np.cumsum(river_dists_new)

        # get point coordinates for all points in Riser instance
        if surface == "lower":
            # approximate using first point in each profile
            x = [xx[0] for xx in self.x]
            y = [yy[0] for yy in self.y]
        elif surface == "upper":
            # approximate using last point in each profile (upper surface)
            x = [xx[-1] for xx in self.x]
            y = [yy[-1] for yy in self.y]
        else:
            raise Exception("surface must be either 'lower' or 'upper'")
        xy_points = list(zip(x,y))

        point_upstream_dist = []
        for i, (x, y) in enumerate(xy_points):
            # calulcate distance between x, y and each river point:
            point_river_dists = \
                np.sqrt((river_x-x)**2 + (river_y-y)**2)
            min_id = np.argmin(point_river_dists)
            #plt.plot(point_river_dists)
            #plt.show()
            point_upstream_dist.append(river_dists[min_id])

        self.upstream_distance = point_upstream_dist
        
        return self


    def resample_raster_along_profiles(
        self,
        rasterpath: str,
        method: str = "linear",
        band_no: int = 1,
        autocenter: bool = True,
        verbose: bool = True,
        savedir: str = ""
    ) -> Self:
        """
        Resample elevation values of all profiles in Riser instance using data
        from a DEM in rasterpath.

        Parameters:
        -----------
            rasterpath: str
                Path to subdirectory containing the DEM in a format accepted
                by ``rasterio.open()``.
            band_no: int
                Raster band to be interpreted as the DEM.
            method: str
                Interpolation method used to extract elevation values from the
                DEM.
            autocenter: bool
                Option to center the new elevations at ``z=0``. True by default.

        Returns:
        --------
            self: Self
                The riser instance.
        """
        if savedir != "":
            # check if savedir exists and load x, y, d, z info from  there.
            savedir_exists = os.path.isdir(os.getcwd()+f"\\{savedir}")

        if savedir != "" and savedir_exists is True:
            loadpath = os.getcwd()+f"\\{savedir}\\"
            files = os.listdir(loadpath)
            csv_names = [filename[:-8] for filename in files]
            dfs = [pd.read_csv(loadpath+file) for file in files]
            self_names = np.array(self.name)
            for i, csv_name in enumerate(csv_names):
                if verbose is True:
                    print(f"Loading elevation values for profile {csv_name}")
                # find out which self.name corresponds to csv_name
                id = np.where(self_names==csv_name)[0][0]
                z_dem = dfs[i]["z"].to_numpy()
                self.z[id] = z_dem
        else:
            z_out = []
            for i, name in enumerate(self.name):
                if verbose is True:
                    print(f"Resampling elevations of profile {name}")
                # get x, y coordinates in rc format:
                rc_points = [(y, x) for x, y in zip(self.x[i], self.y[i])]
                # sample raster
                z_dem = sample_raster_at_points(
                    rasterpath,
                    rc_points,
                    method,
                    band_no
                )
                # center the new elevations
                if autocenter:
                    z_mid = (z_dem.max() + z_dem.min()) / 2
                    #print(z_mid)
                    id_mid = np.where(z_dem<z_mid)[0][-1]
                    z_mid = z_dem[id_mid]
                    z_dem = z_dem - z_mid
                z_out.append(z_dem)

            self.z = z_out

        if savedir != "" and savedir_exists is False:
            os.mkdir(os.getcwd()+f"\\{savedir}")
        if savedir != "":
            # save x, y, d, z in .csv file for faster reloading next time.
            for i, name in enumerate(self.name):
                df = pd.DataFrame({
                    "x": self.x[i],
                    "y": self.y[i],
                    "d": self.d[i],
                    "z": self.z[i]
                })
                df.to_csv(os.getcwd()+f"\\{savedir}\\{name}_DEM.csv")
                
        return self
    
    
    def merge_Riser_instances(
        self,
        riser_instance_list: list
    ) -> Self:
        """
        Appends the information stored in other Riser instances in the Riser instance
        that calls this function. Only affects information stored in list type.
        
        Parameters:
        -----------
            riser_instance_list: list
                List of other Riser instances to be merged into self.
                
        Returns:
        --------
            self: Self
                A Riser instance containing the merged instances.
        """
        for r in riser_instance_list:
            
            # get all attributes of list type
            attr = r.__dict__.items()

            for (key, val) in attr:
                # get base riser attribute
                if type(val) == list:
                    try: 
                        attr_base = getattr(self, key)
                        if type(attr_base) != list:
                            warnings.warn(f"Casting base attribute {key} to list")
                            attr_base = list(attr_base)
                        # get to be added attr
                        attr_r = getattr(r, key)
                        attr_out = attr_base + attr_r
                        setattr(self, key, attr_out)
                    except AttributeError as E:
                        warnings.warn(f"Could not find attribute {key}, ignoring")
 
        return self
        
    def save_Riser_instance_structure(
        self,
        savedir: str = ""
    ):
        """
        Creates a .npy file containing all attributes of the Riser class
        instance. This file can be used via 
        riserfit.load_Riser_instance_structure to re-load a Riser
        class instance without re-running all analysis steps.
        File name will always be 
        self.identifier+"Riser_instance_structure.npy".
        
        Parameters:
        -----------
            savedir: str
                Subdirectory to which the resulting file is saved.
        
        Returns:
        --------
            None
                
        """

        # get all attributes that are not methods, or __NAME__
        attributes = [attr for attr in dir(self) 
              if not attr.startswith('__') 
              and not callable(getattr(self,attr))]

        # use a numpy data array with object dtype
        riser_npy = np.full(len(attributes)*2, np.nan, dtype="object")
        ind = 0
        for attr in attributes:
            riser_npy[ind] = attr
            riser_npy[ind+1] = getattr(self, attr)
            ind += 2

        save_str = (
            os.getcwd() + 
            f"\\{savedir}\\{self.identifier}" + 
            "_Riser_instance_structure.npy"
        )
        
        np.save(
            save_str,
            riser_npy
        )

    def build_Riser_instance_dataframe(
        self
    ) -> pd.DataFrame:
        """
        Creates a pandas dataframe containing data attributes of the Riser 
        class instance. Only contains data that is stored as single string or
        numeric values for each profile. 
        
        Parameters:
        -----------
            None
            
        Returns:
        --------
            df: pd.DataFrame
                Output dataframe.
        """

        # UPDATE THESE IF NEW TYPES ARE INTRODUCED
        # unwanted datatypes:
        no_types = [
            dict, 
            list, 
            np.ndarray, 
            OptimizeResult # from scipy
        ]

        # get all attributes that are not methods, or __NAME__
        attributes = [attr for attr in dir(self) 
              if not attr.startswith('__') 
              and not callable(getattr(self, attr))]
        nrows = len(self.name)
        df = pd.DataFrame()
        # if the attribute contains numbers or strings, save to df

        for attr in attributes:
            col = getattr(self, attr)
            if len(col) == nrows and not (type(col[0]) in no_types): 
                df[attr] = col
            
        return df

    def plot_profile_map(
        self,
        rasterpath: str,
        band_no: int = 1,
        diffusion_type: str = "linear",
        dem_min: float = 300,
        dem_max: float = 410,
        annotate: bool = True
    ):
        """
        Function to plot profiles onto a DEM (must have the same projection!).
        Plot lines are colored according to kt (linear or nonlinear).

        Parameters:
        -----------
            rasterpath: str
                Subdirectory and filename of the DEM
            band_no: int
                Number of band containing elevations. Default is 1.
            diffusion_type: str
                "linear" or "nonlinear", decides the kt to use to color
                profiles on the map.
            dem_min: float
                Lower limit of the color map.
            dem_max: float
                Upper limit of the color map.
            annotate: bool
                Whether to display profile names on the map.

        Returns:
        --------
            None
        """

        raster = rio.open(os.getcwd()+rasterpath)
        dem = raster.read(band_no)
        raster_bounds = raster.bounds
        raster_bounds = [
            raster_bounds[0],
            raster_bounds[2],
            raster_bounds[1],
            raster_bounds[3]
        ]
        fg, ax = plt.subplots(1,1)
        ax.imshow(dem, extent=raster_bounds, cmap="gray",
            vmin=dem_min, vmax=dem_max)
        # choose the kt to use:
        if diffusion_type == "linear":
            kt = np.array(self.best_kt)
        else:
            kt = np.array(self.nonlin_best_t)
        # normalize kt
        norm_kt = (kt-kt.min()) / (kt.max()-kt.min())
        colors = mpl.cm.jet(norm_kt)
        for i, norm_kt in enumerate(norm_kt):
            # plot line:
            ax.plot(self.x[i], self.y[i], c=colors[i])
            # add label with age
            # calculate rotation angle to align with
            if annotate is True:
                angle = \
                    np.arctan((self.y[i][0]-self.y[i][1]) / \
                    (self.x[i][0]-self.x[i][1])) * 180 /np.pi
                ax.text(self.x[i][0], self.y[i][0], f"{kt[i]:.0f}",
                    rotation=angle, horizontalalignment="center",
                    verticalalignment="center"
                )
        # add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap("jet"),
            norm=mpl.colors.Normalize(kt.min(), kt.max())
        )
        sm.set_array([])
        plt.colorbar(sm, label = r"kt$_{lin}$ [m$^2$]" \
            if diffusion_type == "linear" \
            else r"kt$_{nl}$ [m$^2$]")
        plt.show()

##############################################
## Part 3: loading a saved riser instance ##
##############################################

def load_Riser_instance_structure(
    filepath: str,
    identifier: str = ""
) -> Riser:
    """
    This function loads a Riser instance from a ``.npy`` file.
    The Riser attributes saved in the file should be in the form of a
    dict. ``riserfit.save_Riser_instance_structure()``
    outputs a format that can be read by this function.

    Parameters:
    -----------
        filepath: str
            Path to subdirectory and ``.npy`` file.
        identifier: str
            An optional Riser instance identifier,
            set as the Riser.identifier. If "", the identifier
            of the created instance is chosen from 
            the ``.npy`` Riser instance.

    Returns:
    --------
        riser: class
            Instance of the Riser class.

    """

    riser_npy = np.load(
        f"{os.getcwd()}\\{filepath}",
        allow_pickle=True
    )

    # re-form to dict:
    riser_dict = {}
    ind = 0
    while ind < len(riser_npy):
        riser_dict[riser_npy[ind]] = riser_npy[ind+1]
        ind += 2

    inits = ["x", "y", "z", "d", "name", "identifier"]
    riser = Riser(
        riser_dict["x"],
        riser_dict["y"],
        riser_dict["z"],
        riser_dict["d"],
        riser_dict["name"],
        identifier if identifier != "" else riser_dict["identifier"]
    )

    for key, value in riser_dict.items():
        if key not in inits:
            setattr(riser, key, value)
    return riser