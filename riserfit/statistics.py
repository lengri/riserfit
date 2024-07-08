# for type hinting
from __future__ import annotations
from typing import Union, Tuple
from numpy.typing import ArrayLike
from typing_extensions import Self # pre python 3.11

from .profiles import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy as sp
from scipy.interpolate import interpn, interp1d
from scipy.stats import f as f_dist 
from skspatial.objects import Line # reprojecting
from scipy.linalg import block_diag

# small numeric checker
def _is_numeric(value):
    try:
        a = float(value)
        return True
    except:
        return False

def sample_2d_grid(
    p0: float,
    p1: float,
    ax0_coords: ArrayLike,
    ax1_coords: ArrayLike,
    grid: np.ndarray,
    method: str = "linear",
    extrapolate: bool = True
) -> float:
    """
    Uses scipy.interpolate.interpn() to sample an established
    2d grid with respect to two variables,
    like kt and a. Basically a wrapper around interpn().

    Parameters:
    -----------
        p0: float
            Value along first axis (ax0) to be sampled.
        p1: float 
            Value along second axis (ax1) to be sampled.
        ax0_coords: ArrayLike
            Coordinates of std_grid along axis 0. 
        ax1_coords: ArrayLike
            Coordinates of std_grid along axis 1. 
        grid: np.ndarray
            A 2d grid containing floats.
        method: str
            Interpolation method for scipy.interpolate.interpn().
        extrapolate: bool
            Whether to extrapolate from the original dataset.

    Returns:
    --------
        grid_value: float
            The interpolated grid value at the sampled point .
    """

    grid_coords = (np.array(ax0_coords), np.array(ax1_coords))
    sample_point = np.array([p0, p1])

    grid_value = interpn(
        grid_coords, 
        grid, 
        sample_point, 
        method=method,
        bounds_error=(not extrapolate)
    )

    return grid_value[0]


def linear_regression(
    x: ArrayLike,
    y: ArrayLike, 
    weights: ArrayLike = None,
    ignore: list = []
) -> Tuple[float, float, float]:

    """
    Calculates the best fit slope and intercept for a set of
    x and y coordinates, as well as the R2 correlation coefficient.

    Parameters:
    -----------
        x: ArrayLike
            Independent coordinates.
        y: ArrayLike
            Dependent coordinates of length len(x).
        weigths: ArrayLike
            Weights of the data points.
        ignore: list
            List of indices to ignore for linear fitting and
            correlation calculaten (e.g., list of outliers).

    Returns:
    --------
        slope: float
            The slope of the best fit linear function.
        intercept: float
            The y-axis intercept of the best fit linear
            function.
        r2: float
            The R2 correlation coefficient.

    """

    # x, y are lists, reshaped inside function
    # ignore: array of indices to ignore
    x = np.array(x)
    y = np.array(y)

    if len(ignore) > 0:
        x = np.delete(x, ignore)
        y = np.delete(y, ignore)
    if weights == None: # unweighted regression
        weights = np.ones(len(x))

    model = LinearRegression().fit(
        x.reshape((-1, 1)), y, weights
    )
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(x.reshape((-1, 1)), y)
    return (slope, intercept, r2)


def calculate_lin_reg_confidence_interval(
    x: np.array,
    y: np.array,
    slope: float,
    intercept: float,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Confidence interval around linear regression. For a detailed
    discussion of the mathmatics, see chapter 5 of Neter et al., 
    Applied Linear Regression.

    Parameters:
    -----------
        x: np.array
            The independent variable.
        y: np.array
            The dependent variable.
        slope: float
            The slope of the linear regression line.
        intercept: float
            The intercept of the linear regression line.
        alpha: float
            The alpha value of the confidence interval. Must be
            in interval (0, 1).

    Returns:
    --------
        lower_bound: np.array
            Lower uncertainty bound at each y.
        upper_bound: np.array
            Upper uncertainty bound at each y.
    """

    b0 = intercept 
    b1 = slope
    y_hat = b0+b1*x 
    n = len(x)
    x_mean = np.mean(x)
    x_sum = np.sum((x-x_mean)**2)

    MSE = (np.sum((y-y_hat)**2))/(n-2)

    sqrt_f_dist = np.sqrt(
        2*f_dist.ppf(1-alpha, 2, n-2)
    )

    s_y_hat = np.sqrt(MSE*(1/n + ((x-x_mean)**2) / x_sum))

    lower_bound = y_hat - sqrt_f_dist*s_y_hat
    upper_bound = y_hat + sqrt_f_dist*s_y_hat

    return (lower_bound, upper_bound)

# playground function to test influence of noise etc.
class RiserPlayground(Riser):

    """
    This class inherits methods from riserfit.Riser class. 
    It contains additional methods to perfom various Monte Carlo-style
    analyses of noise and other uncertainties. 
    It can copy the data available in a Riser instance, 
    or generate synthetic profiles.
    """

    def __init__( # only initialize all the attributes to be filled
        self, 
        identifier: str = ""
    ) -> None: 
        """
        Initializes a RiserPlayground instance. Because it inherits from
        Riser, any RiserPlayground instance has all attributes initialized
        during riserfit.Riser.__init__().

        Parameters:
        -----------
            identifier: str
                Identifier string of the RiserPlayground instance.

        Returns:
        --------
            None

        """
        super().__init__(
            [],
            [],
            [],  # elevations
            [],  # along profile distances
            [],  # profile name
            identifier # identify string
        )

        #########################################
        # Attributes that are assigned later on #
        #########################################

        # add_gaussian_z_noise() or add_uniform_z_noise()
        self.no_noise_z = []

    def load_profiles_from_Riser(
        self,
        Riser_instance
    ) -> Self:
        """
        Load d, z values from a Riser class instance.

        Parameters:
        -----------
            Riser_instance: class
                An instance of the riserfit.Riser class.

        Returns:
        --------
            self: Self
                The RiserPlayground instance.

        """

        attribute_keys = Riser_instance.__dict__.keys()

        for key in attribute_keys:
            attribute = getattr(Riser_instance, key)
            setattr(self, key, attribute)
            
        return self

    def create_profiles_from_parameters(
        self,
        d: list,
        kt: Union[float, list],
        a: Union[float, list],
        b: Union[float, list], 
        theta: Union[float, list], 
        d_off: Union[float, list] = 0,
        z_off: Union[float, list] = 0,
        name: Union[str, list] = "",
        uniform_d: bool = True,
        use_linear: bool = True,
        max_dt: float = 0.5,
        nonlin_S_c: float = 1.
    ) -> Self:
        """
        Create riser profiles based on user-defined parameters. 
        These can be supplied either as single numbers or lists.

        Parameters:
        -----------
            d: list
                Distance values at which z are to be calculated.
                If uniform_d is False, interpreted as a nested list.
            kt: float | list
                Diffusion age(s) for each profile.
            a: float | list
                Riser height(s) for each profile.
            b: float | list
                Riser far field slope(s) for each profile.
            theta: float | list
                Initial riser slope(s) for each profile.
            d_off: float | list
                Shift(s) applied to d after elevations are calculated.
            z_off: float | list
                Shift(s) applied to z after elevations are calculated.
            name: str | list
                Name(s) for each riser.
            uniform_d: bool
                Defines the role of d: If True, d is interpreted as a 
                single non-nested list or array used as distance values.
                If False, interpreted as a nested list wherein each
                element defines a unique distance array used for creating
                riser profiles.
            use_linear: bool
                If `True`, nterpret kt as linear diffusion age. If `False`,
                assume nonlinear diffusion age. Computing nonlinear profiles
                may take a significant amount of time.
            max_dt: float
                Maximum time step size allowed when building nonlinear profiles.
            nonlin_S_c: float
                Critical slope used for nonlinear riser profiles.

        Returns:
        --------
            self: Self
                The RiserPlayground instance.
        """

        # ignore d for numeric checking
        in_param_dict = {
            "kt": kt,
            "a": a,
            "b": b,
            "theta": theta,
            "d_off": d_off,
            "z_off": z_off,
            "S_c": nonlin_S_c
        }

        # store outputs here
        out_param_dict = {
            "d": [],
            "kt": [],
            "a": [],
            "b": [],
            "theta": [],
            "d_off": [],
            "z_off": [],
            "S_c": [],
            "name": []
        }
        
        # check if the input parameters can be converted to float
        par_numeric = np.array(
            [_is_numeric(p) for p in in_param_dict.values()]
        )
        # get list parameters
        list_params = [
            key for i, key in enumerate(in_param_dict.keys()) 
            if not par_numeric[i]
        ]
        # check if lengths are equal.
        list_lengths = np.array([
            len(in_param_dict[key]) for key in list_params
        ])
        
        # assert that lists have same length:
        try:

            if np.any(list_lengths != list_lengths[0]):
                raise Exception("Input arguments do not have same shapes")

            # some parameters are lists, some aren't -> Cast to same shapes now.
            shape = list_lengths[0]
            
            for key in in_param_dict.keys():
                if key not in list_params:
                    out_param_dict[key] = [in_param_dict[key]]*shape
                else:
                    out_param_dict[key] = in_param_dict[key]
        
        # all parameters must be floats...
        except IndexError as E:

            out_param_dict["d"].append(np.array(d))
            out_param_dict["kt"].append(kt)
            out_param_dict["a"].append(a)
            out_param_dict["b"].append(b)
            out_param_dict["theta"].append(theta)
            out_param_dict["d_off"].append(d_off)
            out_param_dict["z_off"].append(z_off)
            out_param_dict["S_c"].append(nonlin_S_c)
            out_param_dict["name"].append(name)
            
        except Exception as E:
            raise Exception(f"Could not resolve parameters: {E}")
        
        # special cases: names and d
        if uniform_d:
            out_param_dict["d"] = [d]*len(out_param_dict["kt"])
        else:
            if len(list(d)) == len(out_param_dict["kt"]):
                out_param_dict["d"] = list(d)
            else:
                raise Exception("Shape of non-uniform d does not match other parameters")
        
        # deal with names:
        # default: use number progression as name
        if name == "":
            out_param_dict["name"] = [
                "p"+str(i) 
                for i in range(0, len(out_param_dict["kt"]))
            ]
        else:
            # check shape of name
            if len(name) == len(out_param_dict["kt"]):
                out_param_dict["name"] = list(name)
            else:
                raise Exception("Shape of name does not match other parameters")

        # construct profiles
        z_list_out = []
        d_list_out = []
        
        # iterate over all parameters
        zipper = zip(
            out_param_dict["d"],
            out_param_dict["kt"],
            out_param_dict["a"],
            out_param_dict["b"],
            out_param_dict["theta"],
            out_param_dict["z_off"],
            out_param_dict["d_off"],
            out_param_dict["S_c"]
        )
        
        for (d, kt, a, b, theta, zoff, doff, Sc) in zipper:
            
            d_list_out.append(np.array(d) + doff)
            
            # build linear profiles
            if use_linear:
                
                z_list_out.append(
                    analytical_profile(
                        d, kt, a, b, theta
                    ) + zoff
                )
            
            else:
                
                # build up nonlinear profiles
                dx = d[1] - d[0] # assume uniform spacing in d.
                z_init = analytical_profile(d, 0, a, b, theta)
                
                # number of time steps and step size
                n = int(kt / max_dt) # this will give an age smaller than kt.
                dt_last_step = kt - n*max_dt
                
                z_nl, _ = nonlin_diff_perron2011(
                    z_init, dx, max_dt, n, 1, Sc, 2
                )
                
                # calculate last time step
                if dt_last_step > 0:
                    z_nl, _ = nonlin_diff_perron2011(
                        z_nl[-1,:], dx, dt_last_step, 1, 1, Sc, 2 
                    )
                    
                z_list_out.append(z_nl[-1,:] + zoff)
        
        # append results
        self.z = z_list_out 
        self.d = d_list_out         
        
        return self

    def add_gaussian_z_noise(
        self,
        dx: float = 12.,
        cell_shift: float = 0.,
        mean_z: float = 0.,
        std_z: float = 0.75,
        method: str = "linear"
    ) -> Self:
        """
        Add noise specified by a gaussian distribution N(mean_z, std_z)
        to each profile's elevation. If this function is called multiple
        times for the same riserfit.RiserPlayground() instance, noise
        will be cummulative.
        
        Parameters:
        -----------
            dx: float
                Spatial resolution of the gaussian noise. If None,
                a noise value is calculated for each point of data.
                Otherwise, it is calculated at
                np.arange(self.d[i].min()+cell_shift, self.d[i].max(),dx)
                and interpolated to match self.d[i].
            mean_z: float
                The mean of the gaussian distribution.
            std_z: float
                The standard deviation of the gaussian distribution.
            method: str
                The interpolation method used if dx != None.
        
        Returns:
        --------
            self: Self
                The RiserPlayground instance.
        """
        
        for i, (d, z) in enumerate(zip(self.d, self.z)):
            
            # if dx is none, calculate noise directly
            if dx == None:
                z_noise = np.random.normal(
                    mean_z, std_z, size=len(z)
                )
            else:
                # calculate positions for noise
                d_noise = np.arange(d.min()+cell_shift, d.max()+dx, dx)
                # calculate corresponding noise values
                z_noise_dx = np.random.normal(
                    mean_z, std_z, size=d_noise.shape
                )
                intfun = interp1d(
                    d_noise, z_noise_dx, bounds_error=False,
                    kind=method, fill_value=(z_noise_dx[0], z_noise_dx[-1])
                )
                z_noise = intfun(d)
                
            # add noise to original elevations
            self.z[i] = z+z_noise
            
        return self
        
    def best_fit_gaussian_z_noise(
        self,
        dx: float = None, 
        mean_z: Union[float, list] = 0.17,
        std_z: Union[float, list] = 1.28,
        method: str = "linear",
        best_linear_fit_dict: dict = {},
        calc_uncert: bool = False,
        kt_uncert_dict: dict = {"verbose": False}
    ) -> Self:
        """
        Add noise specified by a gaussian distribution N(mean_z, std_z)
        to each profile. If mean_z, std_z may be provided as iterables with
        one entry for each profile. This function can be called multiple
        times in a row and will add noise each time to the original profile
        elevation values.

        Parameters:
        -----------
            dx: float 
                Spatial resolution of the gaussian noise. If None, 
                noise is calculated for each data point. If float,
                noise is calculated along a regular grid and 
                interpolated to the data points.
            mean_z: float | list
                Mean(s) of the gaussian distribution(s) describing the
                level of gaussian noise added to each sample.
            std_z: float | list
                Standard deviation(s) of the gaussian distribution(s)
                describing the noise added to each sample.
            method: str
                Interpolation method used if dx != None.
            best_linear_fit_dict: dict
                Keyword dictionary passed to 
                riserfit.Riser.compute_best_linear_diffusion_fit().
            calc_uncert: bool
                Calculate the kt uncertainties each iteration.
            kt_uncert_dict: dict
                Keywords to pass to calculate_kt_uncertainty().

        Returns:
        --------
            self: Self
                The RiserPlayground instance.
        """

        # check if self.no_noise_z is empty, if not, 
        # use that one as z
        if self.no_noise_z != []:
            self.z = self.no_noise_z.copy()
        else:
            self.no_noise_z = self.z.copy()

        # try converting input mean_z, std_z to float
        # if works, it is a single number -> create filled array
        try:
            mean_z = float(mean_z)
            mean_z = np.ones(len(self.d))*mean_z

        except TypeError:
            if len(mean_z) != len(self.d):
                raise Exception("len(mean_z) not equal to number of profiles")

        # the same for std_z
        try:
            std_z = float(std_z)
            std_z = np.ones(len(self.d))*std_z

        except TypeError:
            if len(std_z) != len(self.d):
                raise Exception("len(mean_z) not equal to number of profiles")

        for i, name in enumerate(self.name):

            if dx == None:
                # calculate noise array of length self.d[i]
                z_noise = np.random.normal(
                    mean_z[i], std_z[i], len(self.d[i])
                )
                
            if dx != None:
                # noise in dx, interpolate at self.d[i]
                dx_array = np.arange(
                    min(self.d[i]), max(self.d[i])+dx, dx 
                )
                z_noise = np.random.normal(
                    mean_z[i], std_z[i], len(dx_array)
                )
                intfun = interp1d(
                    dx_array, z_noise, kind=method, bounds_error=False,
                    fill_value=(z_noise[0], z_noise[-1])
                )
                z_noise = intfun(self.d[i])

            # add noise to z:
            self.z[i] = np.array(self.z[i])+z_noise 

        # call compute_best_linear_diffusion_fit

        self.compute_best_linear_diffusion_fit(**best_linear_fit_dict)
        self.apply_d_z_offsets()
        if calc_uncert: self.calculate_kt_uncertainty(**kt_uncert_dict)

        return self 
    
    def best_nonlin_fit_gaussian_z_noise(
        self,
        dx: float = None, 
        mean_z: Union[float, list] = 0.17,
        std_z: Union[float, list] = 1.28,
        method: str = "linear",
        best_linear_fit_dict: dict = {},
        best_nonlin_fit_dict: dict = {},
        calc_uncert=False
    ) -> Self:
        """
        Add noise specified by a gaussian distribution N(mean_z, std_z)
        to each profile. If mean_z, std_z may be provided as iterables with
        one entry for each profile. If this function is called multiple times
        in sequence, it will add the noise to the original elevations.

        Parameters:
        -----------
            dx: float 
                Spatial resolution of the gaussian noise. If None, 
                noise is calculated for each data point. If float,
                noise is calculated along a regular grid and 
                interpolated to the data points.
            mean_z: float | list
                Mean(s) of the gaussian distribution(s) describing the
                level of gaussian noise added to each sample.
            std_z: float | list
                Standard deviation(s) of the gaussian distribution(s)
                describing the noise added to each sample.
            method: str
                Interpolation method used if dx != None.
            best_linear_fit_dict: dict
                Keyword dictionary passed to 
                riserfit.Riser.compute_best_linear_diffusion_fit()

        Returns:
        --------
            self: Self
                The RiserPlayground instance.

        """

        # check if self.no_noise_z is empty, if not, 
        # use that one as z
        if self.no_noise_z != []:
            self.z = self.no_noise_z.copy()
        else:
            self.no_noise_z = self.z.copy()

        # try converting input mean_z, std_z to float
        # if works, it is a single number -> create filled array
        try:
            mean_z = float(mean_z)
            mean_z = np.ones(len(self.d))*mean_z

        except TypeError:
            if len(mean_z) != len(self.d):
                raise Exception("len(mean_z) not equal to number of profiles")

        # the same for std_z
        try:
            std_z = float(std_z)
            std_z = np.ones(len(self.d))*std_z

        except TypeError:
            if len(std_z) != len(self.d):
                raise Exception("len(mean_z) not equal to number of profiles")

        for i, name in enumerate(self.name):

            if dx == None:
                # calculate noise array of length self.d[i]
                z_noise = np.random.normal(
                    mean_z[i], std_z[i], len(self.d[i])
                )
                
            if dx != None:
                # noise in dx, interpolate at self.d[i]
                dx_array = np.arange(
                    min(self.d[i]), max(self.d[i])+dx, dx 
                )
                z_noise = np.random.normal(
                    mean_z[i], std_z[i], len(dx_array)
                )
                intfun = interp1d(
                    dx_array, z_noise, kind=method, bounds_error=False,
                    fill_value=(z_noise[0], z_noise[-1])
                )
                z_noise = intfun(self.d[i])

            # add noise to z:
            self.z[i] = np.array(self.z[i])+z_noise 

        # call compute_best_linear_diffusion_fit
        self.compute_best_linear_diffusion_fit(**best_linear_fit_dict)
        self.apply_d_z_offsets()
        self.compute_best_nonlinear_diffusion_fit(**best_nonlin_fit_dict)
        if calc_uncert: self.calculate_nonlin_t_uncertainty(verbose=False)

        return self
    
    
    def downsample_upsample_profiles(
        self,
        resample_dx: float = 12.,
        cell_shift: float = 6.,
        method: str = "cubic",
        add_noise: bool = False,
        mean_z: float = 0.,
        std_z: float = 0.75
    ) -> Self:
        """
        Basic simulation of constructing a DEM from averaged binned 
        elevations, and then resampling these binned values back
        to the original resolution. 
        Noise can be added onto the down-sampled DEM resolution cells.        
        
        Parameters:
        -----------
            resample_dx: float
                Grid size of the simulated DEM, in meters.
            cell_shift: float
                Offset at beginning of profile. Prevents
                smaller bins at the beginning and may
                introduce asymmetry in the resampled profile.
                Value should be >= 0, or unaccounted 
                boundary effects may occur.
            method: str
                Interpolation method for resampling. Must be
                accepted by scipy.interpolate.interp1d().
            add_noise: bool
                Option to add noise to the downsampled profile.
            mean_z: float
                Mean value of the vertical noise.
            std_z: float
                Standard deviation of the vertical noise.

        Returns:
        --------
            self: Self
                The RiserPlayground instance.
        """
        
        z_downsample = np.zeros(len(self.z), dtype="object")
        d_downsample = np.zeros(len(self.z), dtype="object")
        # loop that creates an array of downsampled
        # z profiles
        
        for i, (d, z) in enumerate(zip(self.d, self.z)):
            
            min, max = d.min(), d.max()
            
            # get the minimum (shifted) DEM cell center allowed:
            D0 = min + resample_dx/2 + cell_shift

            # construct all cell centers
            d_DEM = np.arange(
                D0, max-resample_dx/2+cell_shift, resample_dx
            )
            
            # construct the partitioned matrix M, which contains the
            # downscaling kernel...
            # | K                    |
            # |     K            0   |
            # |         K            |
            # |    0        ...      |
            # |                    K |
            # each K may be differently sized and depends on the 
            # number of points around each d_DEM[i] satisfying
            # d_DEM[i]-resample_dx/2 <= d_i < d_DEM[i]+resample_dx/2.
            
            # for each d_DEM, calculate the kernel size...
            K_sizes = np.zeros(d_DEM.shape, dtype="int") # array of m

            z_valid_packed = [] # has to be expanded dynamically, since
            # size is not known beforehand.
            
            for j, ddem in enumerate(d_DEM):
                id = np.logical_and(
                    (d >= ddem-resample_dx/2),
                    (d < ddem+resample_dx/2),
                )
                K_sizes[j] = len(d[id])
                z_valid_packed.append(z[id])
                
            z_valid = np.array(
                [i for z_packed in z_valid_packed for i in z_packed]
            )

            K_list = [np.full((m,m), 1/m) for m in K_sizes]
            M = block_diag(*K_list)
            
            # perform convolution and take only one elevation value per cell
            # (does not matter which one we take...)
            z_DEM_dense = np.matmul(M, z_valid)
            
            # extract exactly one elevation value per cell...
            cell_single_id = np.concatenate(
                ([0], np.cumsum(K_sizes[:-1])),
                dtype="int"
            )
            z_DEM = z_DEM_dense[cell_single_id]               

            z_downsample[i] = z_DEM
            d_downsample[i] = d_DEM
            
            # if noise is to be added, do it here
            if add_noise:
                noise_z = np.random.normal(
                    loc=mean_z, scale=std_z, size=len(d_DEM)
                )
                z_downsample[i] += noise_z

        # now, up-sample the profile back to its original resolution.
        z_upsample = np.zeros(len(self.z), dtype="object")

        for i, (d_DEM, z_DEM, d) in enumerate(zip(d_downsample, z_downsample, self.d)):
            intfun = interp1d(
                d_DEM, z_DEM, kind=method, bounds_error=False,
                fill_value=(z_DEM[0], z_DEM[-1])
            )
            z_upsample[i] = intfun(d)
        
        self.z = list(z_upsample)
        
        return self

def calculate_confidence_interval(
    x: np.array, 
    p: np.array, 
    lb: float = 0.158, 
    ub: float = 0.840,
    mode_center: bool = True
) -> Tuple[float, float]:
    """
    Calculates a confidence interval based on a non-parametric KDE.
    If mode_center == False, lb is interpreted as
    integral_{-infty}^lb_x KDE = lb, and ub as 
    integral_{-infty}^ub_x KDE = ub.
    If mode_center == True, lb defines the area under the KDE between 
    lb_x and the mode. ub defines the area between the mode and ub_x.

    Parameters:
    -----------
        x: np.array
            x values corresponding to known KDE values.
        p: np.array
            KDE values.
        lb: float
            Either lower bound of the cumulative KDE, or the area between
            lb_x and mode.
        ub: float
            Either upper bound of the cumultive KDE, or the area between
            mode and ub_x.
        mode_center: bool
            If mode_center == False, lb is interpreted as
            integral_{-infty}^lb_x KDE = lb, and ub as 
            integral_{-infty}^ub_x KDE = ub.
            If mode_center == True, lb defines the area under the KDE 
            between lb_x and the mode. ub defines the area between 
            the mode and ub_x.

    Returns:
    --------
        lb_x: float
            The lower bound.
        ub_x: float
            The upper bound.
    """


    dx = np.zeros(len(x))
    dx[:-1] = x[1:]-x[:-1]
    cpdf = np.cumsum(p*dx)

    if mode_center is True:

        id_mode = np.argmax(p)
        cpdf_mode = cpdf[id_mode]

        id_lb = np.where(cpdf<=(cpdf_mode-lb))[0]

        if list(id_lb) == []: # ensure that bound is >= 0.
            id_lb = np.where(x>0)[0][0]
        else:
            id_lb = id_lb[-1]

        id_ub = np.where(cpdf>=(cpdf_mode+ub))[0]

        lb_x = x[id_lb]
        ub_x = x[id_ub]
    
    else:

        lb_x = x[np.where(cpdf<=lb)[0][-1]]
        ub_x = x[np.where(cpdf<=ub)[0][-1]]

    return (lb_x, ub_x)





    
    
    