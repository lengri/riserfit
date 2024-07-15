# type hints
from __future__ import annotations # necessary for some typing shenanigans
from numpy.typing import ArrayLike
from typing import Union, Tuple, List

# Data analysis
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# internal imports
from .diffusion import nonlin_diff_perron2011
from .riser_maths import (
    analytical_profile, 
    compute_misfit, 
    _transform_mse
)

###################################################
# Linear inversion stuff ##########################
###################################################
    
def _compute_misfit_for_optimization(
    params: np.ndarray,
    d_emp: np.ndarray,
    z_emp: np.ndarray,
    scales: list[float],
    warning_eps: float
) -> float:
    """
    Use ``compute_misfit()`` instead. Only
    difference is the data entry format. This function is used to conform to
    ``scipy.optimize.minimize``, but less user friendly
    compared to ``compute_misfit()``.

    Parameters:
    -----------
        params: np.ndarray
            Array containing, in order, kt, d_off, z_off, a and b.
            All are scaled to one.
        d_emp: np.ndarray
            Along-profile distances.
        z_emp: np.ndarray
            Elevations of the profile.
        theta: float
            Initial riser gradient at midpoint and age zero.
        scales: np.ndarray
            The actual values for parameters listed in params.

    Returns:
    --------
        misfit: float
            Misfit value, calculated by compute_misfit().
    """

    misfit = compute_misfit(
        kt=scales[0]*params[0],
        d_emp=d_emp,
        z_emp=z_emp,
        a=scales[3]*params[3],
        b=scales[4]*params[4],
        theta=scales[5]*params[5],
        d_off=scales[1]*params[1],
        z_off=scales[2]*params[2],
        warning_eps=warning_eps
    )

    return misfit

def _linear_kt_uncertainty_mse(
    kt: float, 
    a: float, 
    b: float,
    theta: float, 
    d: np.ndarray, 
    z: np.ndarray, 
    mse_cutoff: float,
    float_multiplier: float = 1e6
) -> float:
    """
    Used to calculate the MSE manipulated in such a way
    that min(MSE) = [lower_kt, upper_kt], i.e., the 
    uncertainty bounds of kt, assuming linear diffusion.
    This is achieved using the transformation
    MSE = (MSE-mse_cutoff)**2.
    
    Parameters:
    -----------
        kt: float
            Linear diffusion age.
        a: float
            Riser height.
        b: float
            Riser far field slope.
        theta: float
            Initial slope.
        d: np.ndarray
            Relative distances.
        z: np.ndarray
            Relative elevations.
        mse_cutoff: float
            Cutoff for the kt interval as defined in 
            Wei et al. (2015): MSE < MSE_min + sigma.
        float_multiplier: float
            Prevents floating point errors in scipy optimize (?).
    
    Returns:
    --------
        mse_out: float
            The transformed MSE.
            
    """
    # weights
    prof_len = d[-1] - d[0]
    f_i = np.zeros(len(z))
    f_i[1:] = (d[1:] - d[:-1]) / prof_len
    f_i *= (len(f_i)/np.sum(f_i))
    
    # best profile
    z_best = analytical_profile(
        d, kt, a, b, theta
    )
    
    # calculate mse (without root!)
    mse = np.sum(f_i*((z-z_best)**2))/len(z)
    
    # shift down such that misfit is zero if mse = MSE_cutoff
    # ensure that 0 is local minimum. 
    mse_out = ((mse-mse_cutoff)**2) * float_multiplier + 1

    return mse_out


def _linear_kt_uncertainty_mse(
    kt: float,
    d: np.ndarray,
    z: np.ndarray,
    geom_params: dict,
    sigma: float,
    min_mse: float
) -> Tuple[float, float]:
    """
    Transformed MSE for linear uncertainty inversion.
    
    Parameters:
    -----------
        kt: float
            Evaluation diffusion time.
        d: np.ndarray
            Measured distances.
        z: np.ndarray
            Measured elevations.
        geom_params: dict
            a, b, theta of the analytical profile.
            
    Returns:
    --------
        mse_out: float
            The outout transformed MSE.
    """
    z_at_kt = analytical_profile(d, kt, **geom_params)
    mse_out = _transform_mse(d, z, z_at_kt, sigma, min_mse)
    return mse_out 

def _lin_invert_uncertainty(
    d: np.ndarray, 
    z: np.ndarray, 
    kt_best: float, 
    sigma: float, 
    min_mse: float, 
    dt: float = 0.5, 
    max_iteration: int = 10_000,
    geom_params: dict = {},
) -> Tuple[float, float]:
    """
    Find the lower and upper uncertainty bounds for the nonlinear diffusion
    age via inversion.
    
    Parameters:
    -----------
        d: np.ndarray
            Distances of measured profile.
        z: np.ndarray 
            Elevations of the measured profile.
        nonlin_d: np.ndarray
            Distances for the nonlinear modelled profile.
        kt_best: float 
            Age of the best fit profile.
        sigma: float,
            Wei et al. (2015) cutoff criterion. 
        min_mse: float
            Misfit of the best fit profile.
        dt: float = 0.5 
            Time step size.
        max_iteration: int = 10_000
            Number of iterations when searching for the upper bound.
            If `max_iteration` is exceeded, the maximum possible uncertainty
            is at `t_best + dt*max_iteration`
        geom_params: dict = {}
            Dictionary of geometrical parameters for the initial modelled
            profile. (kt, a, b, theta)

    Returns:
    --------
        lower_opt.x[0]: float
            Lower bound for kt.
        upper_opt.x[0]: float
            Upper bound for kt.
    """

    # search for kt uncertainty bounds...
    kt_lb = kt_best
    kt_ub = kt_best
    
    found_ub = False
    found_lb = False
    
    z_reference = analytical_profile(
        d, **geom_params
    )
    mse_reference = _transform_mse(d, z, z_reference, sigma, min_mse)
    n = 0
    
    # iterate through possible constraints until a higher MSE
    # is found.
    while (n < max_iteration) and not (found_ub and found_lb):
  
        # check lower bound...
        if not found_lb:
            
            kt_lb = kt_best - n*dt
            
            if kt_lb <= 0:
                kt_lb = 0
                found_lb = True
            else:
                geom_params["kt"] = kt_lb
                z_at_kt = analytical_profile(d, **geom_params)
                mse_lb = _transform_mse(d, z, z_at_kt, sigma, min_mse)
            
                if (mse_lb > 2*mse_reference):
                    found_lb = True

        # check upper bound 
        if not found_ub:
            kt_ub = kt_best + n*dt
            geom_params["kt"] = kt_ub 
            z_at_kt = analytical_profile(d, **geom_params)
            mse_ub = _transform_mse(d, z, z_at_kt, sigma, min_mse)
            
            if (mse_ub > 2*mse_reference):
                found_ub = True

        n += 1
    
    inv_params = {
        "a": geom_params["a"],
        "b": geom_params["b"],
        "theta": geom_params["theta"]
    }
    args = (
        d, z, inv_params, sigma, min_mse
    )
    
    lower_opt = sp.optimize.minimize(
        fun=_linear_kt_uncertainty_mse,
        x0=np.array([(kt_lb+kt_best)/2]),
        args=args,
        method="Powell",
        bounds=((kt_lb, kt_best),)
    )
    
    upper_opt = sp.optimize.minimize(
        fun=_linear_kt_uncertainty_mse,
        x0=np.array([(kt_best+kt_ub)/2]),
        args=args,
        method="Powell",
        bounds=((kt_best, kt_ub),)
    )

    return (lower_opt.x[0], upper_opt.x[0])

################################################
##### Nonlinear inversion for uncertainty ######
################################################

def _nonlin_transform_mse_at_t(
    t: float, 
    z_nl: np.ndarray, 
    t_nl: np.ndarray, 
    nonlin_d: np.ndarray, 
    S_c: float, 
    d: np.ndarray, 
    z: np.ndarray, 
    sigma: float, 
    min_mse: float
) -> float:
    """
    Calculates the transformed MSE for a specific time
    t assuming nonlinear diffusion. This is used in the
    inversion scheme.
    
    Parameters:
    -----------
        t: float 
            Evaluation time.
        z_nl: np.ndarray 
            An array of `z_nl.shape = (N, M)` where `N` indicates the
            number of time steps and `M` indicates the number of points
            in each profile.
        t_nl: np.ndarray 
            An array of time steps of shape `t_nl.shape = (N, )`.
        nonlin_d: np.ndarray 
            The spacings of profiles with shape `nonlin_d.shape = (M, )`.
        S_c: float
            The critical slope in m/m.
        d: np.ndarray
            The distances of the measured elevations with shape
            `d.shape = (K, )`.
        z: np.ndarray
            The measured elevations with shape `z.shape = (K, )`.
        sigma: float 
            The Wei et al. (2015) cutoff criterion.        
        min_mse: float
            The MSE of the best fit profile.
                
    Returns:
    --------
        mse_out: float
            The transformed MSE at time `t`.
    """
    
    # prevent negative t
    if t < 0: return 999.
    
    # get last time step before requested t
    if np.any(t_nl==t):
        
        id = np.where(t_nl==t)[0][0]
        z_at_t = z_nl[id,:]
    
    # if that does not exist, find closest previous step
    else:
        id = np.where(t_nl<t)[0][-1]
        dt = t - t_nl[id]
        dx = nonlin_d[1]-nonlin_d[0]
        
        # diffusion time step
        z_one, _ = nonlin_diff_perron2011(
            z_nl[id,:], dx=dx, dt=dt, n_t=1,
            k=1, S_c=S_c, n=2
        )

        z_at_t = z_one[-1,:]

    # interpolate to correct spacing
    intfun = sp.interpolate.interp1d(nonlin_d, z_at_t, kind="cubic")
    z_model_int = intfun(d)
    
    mse_out = _transform_mse(d, z_model_int, z, sigma, min_mse)

    return mse_out

def _nonlin_invert_uncertainty(
    d: np.ndarray, 
    z: np.ndarray, 
    nonlin_d: np.ndarray,
    z_best: np.ndarray,
    t_best: float, 
    sigma: float, 
    min_mse: float, 
    S_c: float, 
    dt: float = 0.5, 
    max_iteration: int = 10_000,
    geom_params: dict = {},
    k: float = 1,
    n: int = 2,
    warning_eps: float = np.nan
) -> Tuple[float, float]:
    """
    Find the lower and upper uncertainty bounds for the nonlinear diffusion
    age via inversion.
    
    Parameters:
    -----------
        d: np.ndarray
            Distances of measured profile.
        z: np.ndarray 
            Elevations of the measured profile.
        nonlin_d: np.ndarray
            Distances for the nonlinear modelled profile.
        t_best: float 
            Age of the best fit profile.
        sigma: float,
            Wei et al. (2015) cutoff criterion. 
        min_mse: float
            Misfit of the best fit profile.
        S_c: float 
            Critical slope in m/m.
        dt: float = 0.5 
            Time step size.
        max_iteration: int = 10_000
            Number of iterations when searching for the upper bound.
            If `max_iteration` is exceeded, the maximum possible uncertainty
            is at `t_best + dt*max_iteration`
        geom_params: dict = {}
            Dictionary of geometrical parameters for the initial modelled
            profile. (a, b, theta)
        k: float = 1
            The diffusivity constant.
        n: int = 2
            The exponent of the (S/S_c) term in the nonlinear transport
            equation.
        warning_eps: float = np.nan
            Raise a warning if calculated slopes are less than this value.
            This can be used to detect instability in the numerical solution.
    
    Returns:
    --------
        lower_opt.x[0]: float
            Lower bound.
        upper_opt.x[0]: float
            Upper bound.
    """

    z_init = analytical_profile(
        nonlin_d, **geom_params
    )
    n_t = int(t_best / dt) + 1
    
    z_nl, t_nl = nonlin_diff_perron2011(
        z_init=z_init,
        dx=nonlin_d[1]-nonlin_d[0],
        dt=dt,
        n_t=n_t,
        S_c=S_c,
        n=n,
        k=k,
        warning_eps=warning_eps
    )

    # starting from the last step in t_nl,
    # continuously calculate the transformed mse
    # until MSE at new step is larger than the last one.
    # (i.e. it must be after the local minimum in the
    # transformed mse)

    # search for kt uncertainty bounds...
    t_lb = t_best
    t_ub = t_best
    
    found_ub = False
    found_lb = False
    
    z_reference = z_best
    mse_reference = _transform_mse(d, z, z_reference, sigma, min_mse)
    iteration = 1
    
    # iterate through possible constraints until a higher MSE
    # is found.
    while (iteration <= max_iteration) and not (found_ub and found_lb):

        if not found_lb:
            
            t_lb = t_best - iteration*dt
            
            if t_lb <= 0:
                t_lb = 0
                found_lb = True
            else:
                # get entry from z_nl...
                id = np.where(t_nl < t_lb)[0][-1]
                intfun = sp.interpolate.interp1d(nonlin_d, z_nl[id,:], kind="cubic")
                z_at_t = intfun(d)
                mse_lb = _transform_mse(d, z, z_at_t, sigma, min_mse)
            
                if (mse_lb > 10*mse_reference):
                    found_lb = True

        # check upper bound 
        if not found_ub:
            
            z_onestep, _ = nonlin_diff_perron2011(
                z_nl[-1,:], dx=nonlin_d[1]-nonlin_d[0],
                dt=dt, n_t=1, S_c=S_c, n=n, k=k, warning_eps=warning_eps
            )
            # cat to z_nl and t_nl:

            z_nl = np.concatenate((z_nl, np.array([z_onestep[-1,:]])), axis=0)
            t_nl = np.concatenate((t_nl, [t_nl[-1]+dt]), axis=0)
            
            intfun = sp.interpolate.interp1d(nonlin_d, z_nl[-1,:], kind="cubic")
            z_at_t = intfun(d)
            mse_ub = _transform_mse(d, z, z_at_t, sigma, min_mse)
            
            t_ub = t_nl[-1]
            
            if (mse_ub > 10*mse_reference):
                found_ub = True

        iteration += 1

    # after the loop, z_nl will contain all profiles within the error range.
    # feed this information to the actual inversion algorithm.
    
    # two parts, lower and upper uncertainty bound
    
    # first part
    id_lower = np.where(t_nl <= t_best+dt)[0]
    
    args = (
        z_nl[id_lower,:], t_nl[id_lower], 
        nonlin_d, S_c, d, z, sigma, min_mse
    )
    
    lower_opt = sp.optimize.minimize(
        fun=_nonlin_transform_mse_at_t,
        x0=np.array([(t_lb+t_best)/2]),
        args=args,
        method="Powell",
        bounds=((t_lb, t_best),)
    )
    
    # second part
    id_upper = np.where(t_nl >= t_best-dt)[0]

    args = (
        z_nl[id_upper,:], t_nl[id_upper], 
        nonlin_d, S_c, d, z, sigma, min_mse
    )
    
    upper_opt = sp.optimize.minimize(
        fun=_nonlin_transform_mse_at_t,
        x0=np.array([(t_best+t_ub)/2]),
        args=args,
        method="Powell",
        bounds=((t_best, t_ub),)
    )

    return (lower_opt.x[0], upper_opt.x[0])