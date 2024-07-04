# type hints
from __future__ import annotations
from typing import Union, Tuple
from numpy.typing import ArrayLike
from typing import Callable
from typing_extensions import Self # pre python 3.11

#system stuff
import os, warnings, sys
import warnings

# data analysis, managing and calculations
import numpy as np
import pandas as pd
from skspatial.objects import Line # reprojecting
import matplotlib.pyplot as plt 

# imports for (non-)linear diffusion fitting
import scipy as sp
from scipy.interpolate import interp1d
from scipy.special import erf

# small numeric checker
def _is_numeric(value):
    if type(value) == np.ndarray: return False 
    try:
        a = float(value)
        return True
    except:
        return False

#####################################
## Part 1: Geometries and profiles ##
#####################################

# profile distance calculator, because this appears way too often 
# in the code....
def xy_distances(
    x: ArrayLike, 
    y: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Small function to compute the euclidian distance between equal
    length arrays of x and y coordinates.

    Parameters:
    -----------
        x: ArrayLike
            Coordinates in x direction.
        y: ArrayLike
            Coordinates in y direction.

    Returns:
    --------
        d: np.ndarray
            Cumulative distances starting at 0.
        dd: np.ndarray
            Differential distances.
    """

    x = np.array(x)
    y = np.array(y)

    # check that x, y are equal length:
    if len(x) != len(y):
        raise Exception("x, y do not have same dimensions!")

    # calculate differential distances
    dd = np.zeros(len(x))

    # vectorized euclidian distances
    dd[1:] = np.sqrt(
        (x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2
    )

    # cumulative sum
    d = np.cumsum(dd)

    return (d, dd)


def least_square_reproject(
    x: ArrayLike,
    y: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes best fit line and reprojects all points onto that
    line.

    Parameters:
    -----------
        x: ArrayLike
            Coordinates in x direction.
        y: ArrayLike
            Coordinates in y dirextion.

    Returns:
    --------
        x_out: np.ndarray
            Reprojected coordinates in x direction.
        y_out: np.ndarray
            Reprojected coordinates in y direction.
        d_out: np.ndarray
            Re-calculated distances for the new coordinates.
        dd_out: np.ndarray
            Re-calculated diff. distances for the new
            coordinates.
    """

    x = np.array(x)
    y = np.array(y)

    a, b = np.polyfit(x, y, 1)

    P0 = [x.min(), a * x.min() + b]
    P1 = [x.max(), a * x.max() + b]

    line = Line.from_points(point_a=P0, point_b=P1)
    projX = []
    projY = []

    for ii, _ in enumerate(x):

        projPoint = line.project_point((x[ii], y[ii]))
        projX.append(projPoint[0])
        projY.append(projPoint[1])

    x_out = projX
    y_out = projY

    # re-calculate distances:
    d_out, dd_out = xy_distances(x_out, y_out)

    return (x_out, y_out, d_out, dd_out)



def analytical_profile(
    d: np.ndarray,
    kt: float,
    a: float,
    b: float,
    theta: float,
    warning_eps: float = -1e-10
) -> np.ndarray:
    """
    Function to compute elevation of a riser profile
    defined by kt, a, b and theta along points x. Uses the equation proposed by
    Hanks and Andrews (1989), based on linear diffusion assumptions.

    Parameters:
    -----------
        d: np.ndarray
            Points where the elevations of the scarp are to be calculated
        kt: float
            Diffusion age of the scarp.
        a: float
            Half the initial riser height (at kt=0).
        b: float
            Far field slope of the scarp.
        theta: float
            Initial riser gradient (at kt=0).

    Returns:
    --------
        h: np.ndarray
            Array of elevations for the riser profile.

    """

    # TODO: Add check for very low values? maybe defined by epsilon?
    if type(kt) == list:  # scipy seems to pass type(kt) == list sometimes?
        kt = kt[0]

    if kt < 0:
        if kt < warning_eps:
            warnings.warn(f"Warning: kt = {kt} < 0 returns np.nan!")
        return np.full(len(d), np.nan)

    frac = a / (theta - b)

    if kt == 0:  # erf(np.Inf) = 1

        # need to check where x - frac or x + frac are greater
        # or smaller than zero
        first = (d + frac) > 0
        second = (d - frac) > 0

        # convert to signs, because sp.erf(+-np.Inf) = +-1
        erf1 = first * 2 - 1
        erf2 = second * 2 - 1

        # first part, np.exp(-np.Inf) = 0 -> omit

        h = ((theta - b) / 2) * ((d + frac) * erf1 - (d - frac) * erf2) + b * d

    else:

        # error function part of equation

        erfs = (d + frac) * erf((d + frac) / np.sqrt(4. * kt)) - \
               (d - frac) * erf((d - frac) / np.sqrt(4. * kt))

        # exponential part of equation
        exps = np.exp(-((d + frac) ** 2) / (4. * kt)) - np.exp(-((d - frac) ** 2) / (4. * kt))

        # final equation
        h = (theta - b) * np.sqrt(kt / np.pi) * exps + \
            ((theta - b) / 2) * erfs + b * d

    return h


def analytical_derivative(
    d: np.ndarray,
    kt: float,
    a: float,
    b: float,
    theta: float
) -> np.ndarray:
    """
    analytival_derivative: Function to compute the first derivative in space of
    the riser scarp equation (analytical_profile()).
    Taken from Pelletier et al. (2006).

    Parameters:
    -----------
        d: np.ndarray
            Along-profile distances.
        kt: float
            Diffusion age.
        a: float
            Initial half riser height.
        b: float
            Far field gradient.
        theta: float
            Initial gradient at midpoint and age zero.

    Returns:
    --------
        dhdx: np.ndarray
            Derivative in space for each point.

    """

    frac = a / (theta - b)
    theta_b = theta - b

    if kt < 0:  # this should never happen on purpose, but does during optimize.
        return np.full(len(d), -9999)

    elif kt == 0:  # erf(np.Inf) = 1

        # need to check where x - frac or x + frac are greater
        # or smaller than zero
        first = (d + frac) > 0
        second = (d - frac) > 0

        # convert to signs, because sp.erf(+-np.Inf) = +-1
        erf1 = first * 2 - 1
        erf2 = second * 2 - 1

        dh = (theta_b / 2) * (erf1 - erf2) + b
    else:

        sqrt_kt = np.sqrt(4 * kt)

        dh = (theta_b / 2) * (erf((d + a / theta_b) / sqrt_kt) -
                              erf((d - a / theta_b) / sqrt_kt)) + b

    return dh


def compute_misfit(
    kt: float,
    d_emp: np.array,
    z_emp: np.array,
    a: float,
    b: float,
    theta: float,
    d_off: float = 0,
    z_off: float = 0,
    warning_eps: float = -1e-10
) -> float:
    """
    compute_misfit: Function to compute the misfit between a profile defined
    by d_emp and z_emp and an analytical curve defined by kt, a, b, theta at
    points d_emp. The function uses the weighted misfit formula
    of Avouac (1993). d_off and z_off are set to zero by default, but should
    be varied during curve fitting.

    Parameters:
    -----------
        kt: float
            Diffusion age of the analytical profile.
        d_emp: np.array
            Along profile distances for a profile.
        z_emp: np.array
            Elevations along the profile.
        a: float
            Half of the initial scarp height of the analytical profile.
        b: float
            Far field gradient of the analytical profile.
        theta: float
            Initial gradient of the analytical profile.
        d_off: float
            Offset in along-profile distance, used to adjust riser profile in
            horizontal direction.
        z_off: float
            Offset in elevation, used to adjust riser profile in vertical
            direction.

    Returns:
    --------
        misfit: float
            Misfit value indicating the quality of fit between the real and
            analytical profile
    """

    # compute analytical solution (offset according to d_off, z_off)

    z_ana = analytical_profile(d_emp + d_off, kt, a, b, theta, warning_eps)
    z_emp = z_emp + z_off

    # compute weights

    prof_len = d_emp[-1] - d_emp[0]
    f_i = np.zeros(len(d_emp))
    f_i[1:] = (d_emp[1:] - d_emp[:-1]) / prof_len
    f_i /= np.sum(f_i)
    f_i *= len(d_emp) # this is not as in Avouac 1993, but makes it comparable to unweighted RMSE
    
    # compute misfit
    rmse = np.sqrt(np.sum(f_i * ((z_ana - z_emp) ** 2)) / len(z_ana))

    return rmse


def distance_along_line(
    x: float,
    y: float,
    point_list: list
) -> int:
    """
    Caluclates the minimum distance between the provided point (x, y), and 
    a list of point tuples (xx, yy), and returns the the id of the closest tuple.
    
    Parameters:
    -----------
        x: float
            X coordinate.
        y: float
            Y coordinate.
        point_list: list
            List of points in (x, y) format.
            
    Returns:
    --------
        id_min: int
            Index of closest tuple in point_list.
        
    """
    
    dists = [
        np.sqrt((xx-x)**2+(yy-y)**2) for (xx, yy) in point_list
    ]
    
    id_min = np.argmin(dists)
    if not _is_numeric(id_min):
        id_min = id_min[0]
    return id_min

##############################################
## PART 2: Statistics to calculate kt, k, t ##
##############################################

def calculate_wei_2015_sigma(
    residuals: np.ndarray
) -> float:
    """
    Calculates the sigma used for uncertainty interval calculation
    in Wei et al. 2015.

    Parameters:
    -----------
        residuals: np.ndarray
            The residuals between data and model.

    Returns:
    --------
        sigma: float
            The sigma as in Wei et al. 2015.
    """

    sigma_res = np.std(residuals, ddof=1)
    sigma = np.sqrt((2*sigma_res**4)/(len(residuals)-1))

    return sigma


def calculate_function_jacobian(
    func: Callable,
    x_values: ArrayLike,
    parameter_values: dict,
    epsilon: Union[ArrayLike, float] = 1.4901161193847656e-08
) -> np.ndarray:
    """
    Calculates a finite difference approximation of the Jacobian.
    If func takes p+1 parameters, the Jacobian will have p columns.
    The first function argument must always be a vector of x values.
    For n x_values, the Jacobian will have n rows.

    Parameters:
    -----------
        func: Callable
            Function of the form func(x_values, ...).
        x_values: ArrayLike
            x values at which the partial derivatives are
            evaluated.
        parameter_values: dict
            Dictionary of the form {key: value, ...} for all
            parameters to be passed on to func(), except
            x_values. The partial derivatives are calculated,
            in the same order, for each key in the dict.
        epsilon: ArrayLike | float
            Epsilon used for finite difference derivative 
            calculation. May be an array of length
            len(parameter_values.keys()) or a single float.

    Returns:
    --------
        jacobian: np.ndarray
            The Jacobian matrix of func().
    """

    jacobian = np.zeros(
        (len(x_values), len(parameter_values.keys()))
    )

    for i, _ in enumerate(parameter_values.keys()):

        # parameter dict is adjusted for each partial
        # derivative.
        params1 = parameter_values.copy()
        params2 = parameter_values.copy()
        if _is_numeric(epsilon):
            params1[list(params1)[i]] += epsilon
            params2[list(params1)[i]] -= epsilon
        else:
            params1[list(params1)[i]] += epsilon[i]
            params2[list(params1)[i]] -= epsilon[i]

        diff1 = func(x_values, **params1)
        diff2 = func(x_values, **params2)

        eps = epsilon if _is_numeric(epsilon) else epsilon[i]
        deriv = (diff2-diff1)/(2*eps)

        jacobian[:,i] = deriv

    return jacobian


def propagate_division_error(
    num: float, 
    denom: float, 
    num_error: float, 
    denom_error: float
) -> float:
    """
    Approximates the error for a fraction num / denom where both
    the numerator and the denominator have an associated error.

    Parameters:
    -----------
        num: float
            Numerator.
        denum: float
            Denominator.
        num_error: float
            Error or uncertainty associated with the numerator.
        denom_error: float
            Error or uncertainty associated with the denominator.

    Returns:
    --------
        error: float
            The propagated error.
    """

    error = np.sqrt(
        (num_error/num)**2 + (denom_error/denom)**2
    )
    return error * (num/denom)


def riser_covariance_matrix(
    jacobian: np.ndarray,
    sum_of_squares: float
) -> np.ndarray:
    """
    Converts a jacobian to a covariance matrix based on 
    :math:`Cov = sos\cdot(J^t \cdot J)^(-1)`
    
    Parameters:
    -----------
        jacobian: np.ndarray
            The Jacobian.
        sum_of_squares: float
            Sum of squares error.
    
    Returns:
    --------
        cov_mat: np.ndarray
            The covariance matrix.
    """

    s2 = sum_of_squares / (jacobian.shape[0]-jacobian.shape[1])
    jac_inv = np.linalg.inv(
        np.matmul(np.transpose(jacobian), jacobian)
    )
    cov_mat = s2 * jac_inv

    return cov_mat


def maximum_combined_kde(
    x_vals: np.ndarray, 
    kdes: list, 
    bounds: Tuple[float, float] = (None, None)
) -> Tuple[float, int]:
    """
    Expects a list of arrays, each of len(x_vals). Will combine the KDEs and
    compute the maximum-likelihood of all KDEs. Returns the x value and id.
    
    Parameters:
    -----------
        x_vals: np.ndarray
            X values at which the KDEs are defined.
        kdes: list
            List of np.ndarrays containing the KDE values.
        bounds: Tuple[float, float]
            Bounds applied to x_vals within which a maximum value is 
            calculated. Should be (lower_bound, upper_bound).
    
    Returns:
    --------
        x_max: float
            Position of the maximum KDE value.
        id_max: int
            ID of the maximum value.
    """
    id_lower = 0
    id_upper = -1
    if bounds != (None, None):
        # expect (float, float)
        id_lower = np.where(x_vals<bounds[0])[0][-1]
        id_upper = np.where(x_vals>bounds[1])[0][0]
        
    kde_2d = np.zeros((len(kdes), len(x_vals)))
    for i, kde_i in enumerate(kdes):
        kde_2d[i,:] = kde_i
    kde_avg = np.nanmean(kde_2d, axis=0)
    id_max = np.argmax(kde_avg[id_lower:id_upper])
    x_max = x_vals[id_lower:id_upper][id_max]
    
    return (x_max, id_max+id_lower)

def gaussian(
    x: np.ndarray, 
    mu: float, 
    sigma: float
) -> np.ndarray:
    """
    Gaussian/Normal distribution.
    
    Parameters:
    -----------
        x: np.ndarray
            Values at which to return PDF values.
        mu: float
            The mean of the Gaussian.
        sigma: float
            The standard deviation of the Gaussian.
    
    Returns:
    --------
        pdf: np.ndarray
            The Gaussian PDF values.
    """
    frac = 1 / (sigma*np.sqrt(2*np.pi))
    exponent = -((x-mu)**2)/(2*sigma**2)
    pdf = frac*np.exp(exponent)
    return pdf

def gaussian_kernel(
    x: np.ndarray, 
    mu: float, 
    sigma: float
) -> np.ndarray: 
    """
    Returns the PDF of a Gaussian distribution
    for an input array x.
    
    Parameters:
    -----------
        x: np.ndarray
            Values along the x axis.
        mu: float
            Mean of the Gaussian distribution.
        sigma: float
            Standard deviation of the 
            Gaussian distribution.
            
    Returns:
    --------
        pdf: np.ndarray
            PDF values of the Gaussian at positions
            specified by x.

    """
    frac = 1 / (sigma*np.sqrt(2*np.pi)) 
    exponent = -((x-mu)**2)/(2*sigma**2)
    pdf = frac*np.exp(exponent)
    return pdf

def triang_kernel(
    x: np.ndarray, 
    lb: float, 
    mid: float, 
    ub: float
) -> np.ndarray:
    """
    A triangular PDF.
    
    Parameters:
    -----------
        x: np.ndarray
            Values along the x axis.
        lb: float
            Lower bound of the PDF.
        mid: float
            Maximum value of the PDF.
        ub: float
            Upper bound of the PDF.
    
    Returns:
    --------
        pdf: np.ndarray
            The triangular PDF values.
    """
    y = 2 / ((mid-lb)+(ub-mid))
    m1 = y / (mid-lb)
    m2 = y / (mid-ub)
    n1 = -lb*m1
    n2 = -ub*m2 
    
    inside1 = np.logical_and(x>lb, x<=mid) 
    inside2 = np.logical_and(x>mid, x<=ub)
    
    pdf = np.zeros(x.shape)
    pdf[inside1] = m1*x[inside1]+n1 
    pdf[inside2] = m2*x[inside2]+n2 
    return pdf

def triang_kde(
    x: np.ndarray, 
    lb: np.ndarray, 
    mid: np.ndarray, 
    ub: np.ndarray
) -> np.ndarray:
    """
    A KDE composed of multiple triangular PDFs.
    
    Parameters:
    -----------
        x: np.ndarray
            Values along the x axis.
        lb: np.ndarray
            Lower bounds of the triangular PDFs.
        mid: np.ndarray
            Maximum values of the triangular PDFs.
        ub: np.ndarray
            Upper bounds of the triangular PDFs.
    
    Returns:
    --------
        pdf: np.ndarray
            The KDE values.
    """
    tot_sum = np.zeros(len(x))
    for (m, l, u) in zip(mid, lb, ub):
        tot_sum += triang_kernel(x, l, m, u)
    return tot_sum/len(mid)

def kde_gaussian(
    x: np.ndarray, 
    xi: np.ndarray, 
    si: np.ndarray
):
    """
    Creates a KDE (or composite PDF) out of multiple
    Gaussian distributions.
    
    Parameters:
    -----------
        x: np.ndarray
            The values along the x axis where KDE
            values are output.
        xi: np.ndarray
            An array of Gaussian mean values (mu)
            for each Gaussian distribution.
        si: np.ndarray
            An array of standard deviations (sigma)
            for each Gaussian distribution.
            
    Returns:
    --------
        kde: np.ndarray
            Array of length len(x) containing KDE
            values along the x axis.

    """
    tot_sum = np.zeros(len(x))
    for mu, s in zip(xi, si):
        tot_sum += gaussian_kernel(x, mu, s)
    kde = tot_sum/len(xi)
    return kde


class StatsMC:
    """
    This class provides tools for Monte Carlo simulations 
    to calibrate diffusivity and morphological age 
    based on a set of kt and an estimate of t.
    """
    def __init__(
        self, 
        kt: Union[list, np.ndarray], 
        lb_kt: Union[list, np.ndarray], 
        ub_kt: Union[list, np.ndarray], 
        t: Union[float, list, np.ndarray], 
        t_sigma: Union[float, list, np.ndarray],
        identifier: str = "StatsMC"
    ):
        """
        Initialise a StatsMC instance.
        
        Parameters:
        -----------
            kt: Union[list, np.ndarray]
                Diffusion ages of the Riser instance.
            lb_kt: Union[list, np.ndarray]
                Lower kt uncertainties.
            ub_kt: Union[list, np.ndarray]
                Upper kt uncertainties.
            t: Union[float, list, np.ndarray] 
                External age constraints of the risers. These may be multiple
                ages, e.g. if multiple ages for a single terrace exist.
            t_sigma: Union[float, list, np.ndarray]
                External age uncertainties.
            identifier: str
                The identifier string of the StatsMC instance.
            
        Returns:
        --------
            None
        """

        self.kt = np.array(kt)  
        self.lb_kt = np.array(lb_kt) 
        self.ub_kt = np.array(ub_kt) 
        self.t = np.array([t]) if _is_numeric(t) else np.array(t) 
        self.t_sigma = np.array([t_sigma]) if _is_numeric(t_sigma) else np.array(t_sigma)

        self.identifier = identifier
        # Additional attributes to be assigned later on
        self.kt_kde = None
        self.initial_t_kde = None 
        self.MC_t_kde = None
        self.k_kde = None
        self.MC_t_sample = None
        
    def construct_kt_kde(
        self,
        kde: callable = triang_kde,
        min_val: float = 1e-5,
        max_val: float = 1e4,
        kt_resolution: float = 1
    ) -> Self:
        """
        Constructs a KDE of kt from which to sample. This self.kt_kde inherits
        its methods from scipy.stats.rv_continuous
        
        Parameters:
        -----------
            kde: callable
                A function that takes on values specified by kde_dict.
                Must take arguments in the form kde(x, lb, mid, ub).
            min_val: float
                Lower bound of the KDE, if not defined within the kde callable.
            max_val: float
                Upper bound of the KDE, if not defined within the kde callable.
                
        Returns:
        --------
            self: Self
                A PDF instance of the riserfit.CustomDistribution class.
        """
        
        # store min and max
        self.kt_min_val = min_val
        self.kt_max_val = max_val
        ktx = np.arange(min_val, max_val+kt_resolution, kt_resolution)
        pdf_kt = kde(ktx, lb=self.lb_kt, mid=self.kt, ub=self.ub_kt)
        
        self.kt_kde = DistributionFromInterpolator(
            x=ktx, pdf=pdf_kt
        )
        
        return self
    
    def construct_initial_t_kde(
        self,
        kde: callable = kde_gaussian,
        t_resolution: float = 1
    ) -> Self:
        """
        Construct a KDE from independent age constraints. This function
        sets the StatsMC.initial_t_kde attribute, which is an instance
        of riserfit.DistributionFromInterpolator.
        
        Parameters:
        -----------
            kde: callable
                The type of KDE to be output. Must be a function.
                Default is riserfit.kde_gaussian.
            t_resolution: float
                The resolution at which t values are calculated.
        
        Returns:
        --------
            self: Self
                The StatsMC instance.
        """
        # store min and max
        self.t_min_val = 1e-5 # should not be 0 due to division error
        id_max = np.argmax(self.t)
        self.t_max_val = self.t[id_max]+self.t_sigma[id_max]*5
        
        tx = np.arange(self.t_min_val, self.t_max_val+t_resolution, t_resolution)
        pdf_t = kde(tx, xi=self.t, si=self.t_sigma)

        self.initial_t_kde = DistributionFromInterpolator(
            x=tx, pdf=pdf_t
        )
        
        return self
    
    def set_k_kde(
        self,
        k_kde: callable
    ) -> Self:
        """
        Use a k_kde from some external source to set StatsMC.k_kde. 
        Should behave similar to 
        a riserfit.DistributionFromInterpolator instance.
        
        Parameters:
        -----------
            k_kde: callable
                KDE to be used for drawing random samples from.
        
        Returns:
        --------
            self: Self
                The StatsMC instance.
        """
        self.k_kde = k_kde
        self.k_min_val = np.min(k_kde.x)
        self.k_max_val = np.max(k_kde.x)
        
        return self
        
    def construct_MC_k_kde(
        self,
        n: int = 10_000,
        min_val: float = 1e-5,
        max_val: float = 1e2,
        k_resolution: float = 0.25
    ) -> Self:
        """
        Estimate a PDF of k by sampling self.kt_kde and self.t_kde. 
        t is treated as a Gaussian variable and the resulting KDE kernel is 
        a reciprocal inverse Gaussian. The resulting KDE is available in
        StatsMC.k_kde and is an instance of riserfit.DistributionFromInterpolator.
        
        Parameters:
        -----------
            n: int
                Number of random draws from self.kt_kde
            min_val: float
                Minimum value for the k PDF.
            max_val: float
                Maximum value for the k PDF.
            
        Returns:
        --------
            self: Self
                A PDF instance of the 
                riserfit.DistributionFromInterpolator class.
        """
        
        # save min and max
        self.k_min_val = min_val 
        self.k_max_val = max_val 
        
        ktx = np.linspace(self.kt_min_val, self.kt_max_val, 10*n)
        tx = np.linspace(self.t_min_val, self.t_max_val, 10*n)

        # create two weighted samples
        pdf_kt = self.kt_kde.pdf(ktx)
        pdf_t = self.initial_t_kde.pdf(tx)
        kt_sample = np.random.choice(
            ktx, size=n, replace=True, p=pdf_kt/np.sum(pdf_kt)
        )

        t_sample = np.random.choice(
            tx, size=n, replace=True, p=pdf_t/np.sum(pdf_t)
        )
        
        # calculate CDF and differentiate to PDF
        k_sample = np.sort(kt_sample / t_sample)
        
        # pass sample to distribution creator
        self.k_kde = distribution_from_sample(
            sample=k_sample,
            resolution=k_resolution,
            bounds=(min_val, max_val)
        )
        self.k_sample = k_sample
        
        return self
    
    ##### EXPERIMENTAL
    def construct_paired_t_kde(
        self,
        min_val = 1e-6,
        max_val = 10_000,
        n_vals=100_000,
        t_resolution=1
    ):
        """
        EXPERIMENTAL; DO NOT USE
        
        Parameters:
        -----------
            None
        
        Returns:
        --------
            None
        """
        warnings.warn("construct_paired_t_kde is experimental and should not be used!")
        # from inverse CDFs, take x values corresponding to quantiles
        q = np.linspace(0.01, 0.99, n_vals)
        k_sample = self.k_kde.inverse_cdf(q)
        kt_sample = self.kt_kde.inverse_cdf(q)
        
        t_vals = np.sort(kt_sample / k_sample)
        # generate a CDF and then PDF from this t sample
        cdf_x = np.linspace(1/n_vals, 1, n_vals)
        tx = np.arange(min_val, max_val+t_resolution, t_resolution)

        cdf_intfun = sp.interpolate.interp1d(
            t_vals, cdf_x, bounds_error=False,
            fill_value=(0, 1)
        )        
        cdf_t = cdf_intfun(tx)
        pdf_t = np.zeros(cdf_t.shape)
        pdf_t[1:-1] = (cdf_t[2:]-cdf_t[:-2]) / (tx[2:]-tx[:-2])
        pdf_t[0] = (cdf_t[0]-cdf_t[1]) / (tx[0]-tx[1])
        pdf_t[-1] = (cdf_t[-1]-cdf_t[-2]) / (tx[-1]-tx[-2])
        # print(np.trapz(y=pdf_k, x=kx))
        self.paired_t_kde = DistributionFromInterpolator(
            tx, pdf_t
        )
        
        return self
        
    ##### EXPERIMENTAL END
    
    def construct_MC_t_kde(
        self,
        min_val: float = 1e-5,
        max_val: float = 10_000,
        n: int = 10_000,
        t_resolution: float = 1
    ) -> Self:
        """
        Generate a set of morphological ages (with unit time) from
        self.k_kde and self.kt_kde. This is independent of self.t_kde!
        The resulting KDE is available in StatsMC.MC_t_kde and is an
        instance of riserfit.DistributionFromInterpolator.
        
        Parameters:
        -----------
            min_val: float
                Lower bound of the KDE.
            max_val: float
                Upper bound of the KDE.
            n: int
                Number of draws.
            t_resolution: float
                Resolution of t values.
        
        Returns:
        --------
            self: Self
                The StatsMC instance.
            
        """
        
        ktx = np.linspace(self.kt_min_val, self.kt_max_val, 10*n)
        pdf = self.kt_kde.pdf(ktx)
        kt_sample = np.random.choice(
            ktx, size=n, replace=True, p=pdf/np.sum(pdf)
        )
        
        kx = np.linspace(self.k_min_val, self.k_max_val, 10*n)
        pdf = self.k_kde.pdf(kx)
        k_sample = np.random.choice(
            kx, size=n, replace=True, p=pdf/np.sum(pdf)
        )

        # k must not be 0...
        k_sample = k_sample[k_sample>0]
        kt_sample = kt_sample[k_sample>0]

        t_sample = kt_sample / k_sample
        
        # pass sample to distribution creator
        self.MC_t_kde = distribution_from_sample(
            sample=t_sample,
            resolution=t_resolution,
            bounds=(min_val, max_val)
        )
        self.MC_t_sample = t_sample
        
        return self     
    
def construct_averaged_kde(
  list_of_StatsMC: list,
  min_val: float = 1e-5,
  max_val: float = 1e2,
  resolution: float = 0.05,
  attr: str = "k_kde",  
) -> DistributionFromInterpolator:
    """
    Computes an averaged PDF based on a number of input PDFs. This function
    is explicitly for use on multiple StatsMC instances.
    
    Parameters:
    -----------
        list_of_StatsMC: list
            List containing StatsMC instances. Each instance must posses
            the PDF or KDE to be sampled.
        min_val: float
            Minimum value of the averaged PDF.
        max_val: float
            Maximum value of the averaged PDF.
        resolution: float
            Step size of the averaged PDF.
        attr: str
            The KDE or PDF to be averaged for each StatsMC. Default is
            ``k_kde``. Must match an existing instance attribute.
            
    Returns: 
    --------
        pdf: DistributionFromInterpolater
            Distribution instance containing the averaged PDF.
            
    """
    x = np.arange(min_val, max_val+resolution, resolution)
    pdf_2d = np.zeros((len(list_of_StatsMC), len(x)))
    for i, smc in enumerate(list_of_StatsMC):
        pdf_2d[i,:] = getattr(smc, attr).pdf(x)
        
    pdf = np.mean(pdf_2d, axis=0)
    
    return DistributionFromInterpolator(x, pdf)

class DistributionFromInterpolator():
    """
    Class that contains basic statistical tools for the use of probability
    density functions.
    This class is mainly for internal use. Some internal functionalities may not be
    entirely correct in a strictly mathematical sense.
    """
    def __init__(
        self, 
        x: np.ndarray, 
        pdf: np.ndarray
    ) -> None:
        """
        Initialize a ``DistributionFromInterpolator`` instance.

        Parameters:
        -----------
            x: np.ndarray
                X values for the PDF.
            pdf: np.ndarray
                Values of the PDF. Must match the shape of x.

        Returns:
        --------
            None
        """
        self.x = x
        self.density=pdf
        self.density /= np.trapz(y=self.density, x=self.x)
        
        self.pdf = sp.interpolate.interp1d(
            self.x, self.density, bounds_error=False,
            fill_value=(0, 0)
        )
        self._min_val = np.min(x)
        self._max_val = np.max(x)
        
        # construct CDF:
        dx = np.zeros(self.x.shape)
        dx[1:-1] = (self.x[2:]-self.x[:-2])/2
        dx[0] = self.x[1]-self.x[0]
        dx[-1] = self.x[-1]-self.x[-2]
        self.cdf = sp.interpolate.interp1d(
            self.x, np.cumsum(self.density*dx),
            bounds_error=False, fill_value=(0, 1)
        )
        
        # construct inverse CDF
        # this fill value only works for well-behaved x!!
        self.inverse_cdf = sp.interpolate.interp1d(
            np.cumsum(self.density*dx), self.x,
            bounds_error=False, fill_value=(0, 1)
        )
    
    def sample(
        self, 
        resolution: float = None, 
        n: int = 10_000
    ) -> np.ndarray:
        """
        Sample values from the PDF.

        Parameters:
        -----------
            resolution: float
                The resolution in x. If ``None``, values are sampled from self.x.
            n: float
                Sample size.

        Returns:
        --------
            sample: np.ndarray
                Sample from the PDF.
        """
        if resolution == None:
            sample = np.random.choice(
                self.x, 
                size=n, 
                replace=True, 
                p=self.density/np.sum(self.density)
            )
        else:          
            x = np.arange(
                self._min_val, self._max_val+resolution, resolution
            )
            dens = self.pdf(x)
            sample = np.random.choice(
                x, 
                size=n, 
                replace=True, 
                p=dens/np.sum(dens)
            )
 
        return sample

def distribution_from_sample(
    sample: np.ndarray,
    resolution: float,
    bounds: tuple = (0, 0)
) -> DistributionFromInterpolator: 
    
    """
    Create a riserfit.DistributionFromInterpolator instance from
    a sample of size n. A PDF is generated by calculating a CDF,
    interpolating it to even spacing determined by `resolution`
    and calculating its derivative, the PDF.
    
    Parameters:
    -----------
        sample: np.ndarray
            The sample array.
        resolution: float
            The resolution at which the CDF is interpolated.
            Smaller values can give more accurate results, but also
            increase the noise that is present.
        bounds: tuple
            Optional bounds within which the PDF has values larger than 0.
            If `(0, 0)`, the minimum and maximum sample values are used as bounds.
            
    Returns:
    --------
        distribution: DistributionFromInterpolator
            A `DistributionFromInterpolator` instance.
    """
    # sort sample and get size
    sample_sorted = np.sort(sample)
    n = len(sample_sorted)
    
    # determine bounds:
    if bounds == (0, 0):
        bounds = (sample_sorted[0], sample_sorted[-1])

    y_cdf = np.linspace(1/n, 1, n)
    x_cdf = np.arange(bounds[0], bounds[1]+resolution, resolution)  
    
    intfun = sp.interpolate.interp1d(
        sample_sorted, y_cdf, bounds_error=False,
        fill_value=(0, 1)
    )
    
    cdf_even = intfun(x_cdf)
    
    # calculate derivative of CDF (PDF)
    pdf = np.zeros(cdf_even.shape)
    pdf[1:-1] = (cdf_even[2:]-cdf_even[:-2]) / (x_cdf[2:]-x_cdf[:-2])
    pdf[0] = (cdf_even[0]-cdf_even[1]) / (x_cdf[0]-x_cdf[1])
    pdf[-1] = (cdf_even[-1]-cdf_even[-2]) / (x_cdf[-1]-x_cdf[-2])
    
    distribution = DistributionFromInterpolator(
        x_cdf, pdf
    )
    
    return distribution
     
