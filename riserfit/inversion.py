# type hints
from __future__ import annotations # necessary for some typing shenanigans
from numpy.typing import ArrayLike
from typing import Union, Tuple, List

# Data analysis
import numpy as np
import scipy as sp

# internal imports
from .diffusion import nonlin_diff_perron2011
from .riser_maths import analytical_profile, compute_misfit

def compute_misfit_for_optimization(
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
    # print(a)
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


def _nonlinear_t_uncertainty_mse(
    t: float,
    d: np.ndarray,
    z: np.ndarray,
    k: float,
    S_c: float,
    n: float,
    d_nl: np.ndarray,
    warning_eps: float,
    z_nl_matrix: np.ndarray,
    t_nl_matrix: np.ndarray,
    mse_cutoff: float,
    float_multiplier: float = 1e6
) -> float:

    """
    Used to calculate the MSE manipulated in such a way
    that min(MSE) = [lower_t, upper_t], i.e., the 
    uncertainty bounds of t, assuming nonlinear diffusion.
    This is achieved using the transformation
    MSE = (MSE-mse_cutoff)**2.
    
    Parameters:
    -----------
        t: float
            Time for which misfit should be evaluated.
        d: np.ndarray
            Relative distances.
        z: np.ndarray
            Relative elevations.
        k: float
            Nonlinear diffusivity.
        S_c: float
            Critical slope.
        n: float
            Exponent of the nonlinear diffusion equation.
        d_nl: np.ndarray
            The distances used for nonlinear diffusion 
            calculations (equally spaced).
        warning_eps: float
            Output warning if slopes are lower than this value.
        z_nl_matrix: np.ndarray
            2d numpy array of pre-computed nonlinear diffusion
            profiles of the form.
        t_nl_matrix: np.ndarray
            Time steps of the previously computed profiles.
        mse_cutoff: float
            Cutoff for t uncertainty interval, as in Wei et al.
            2015: MSE < MSE_min + sigma
        float_multiplier: float
            Prevents floating point errors in scipy optimize (?).
    
    Returns:
    --------
        mse_out: float
            The transformed MSE at time t.
    """
    
    # prevent negative time evals by returning high mse:
    #print(t<0)
    if t < 0: return 999.
    
    # check if t is in t_nl_matrix:
    id = np.where(t_nl_matrix==t)[0]

    if len(id)>0:
        z_best = z_nl_matrix[id[0],:]
    else:
        # get the timestep prior to t:
        id = np.where(t_nl_matrix<t)[0][-1]

        z_start = z_nl_matrix[id,:]
        # run nonlinear diffusion for one time step
        dx = d_nl[1]-d_nl[0]
        dt = t-t_nl_matrix[id]
        z_1step, _ = nonlin_diff_perron2011(
            z_start, dx, dt, 1, k, S_c, n, warning_eps
        )

        z_best = z_1step[-1,:]
    
    # interpolate to match spacing of d
    intfun = sp.interpolate.interp1d(d_nl, z_best, kind="cubic")
    z_at_t = intfun(d)
    
    # calculate transformed misfit

    # weights
    prof_len = d[-1] - d[0]
    f_i = np.zeros(len(z))
    f_i[1:] = (d[1:] - d[:-1]) / prof_len
    f_i *= (len(f_i)/np.sum(f_i))
    
    # calculate mse (without root!)
    mse = np.sum(f_i*((z-z_at_t)**2))/len(z)
    
    # shift down such that misfit is zero if mse = MSE_cutoff
    # ensure that 0 is local minimum. 
    mse_out = ((mse-mse_cutoff)**2) * float_multiplier + 1

    return mse_out


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


def _transform_mse(
    d: np.ndarray, 
    z1: np.ndarray, 
    z2: np.ndarray, 
    sigma: float,
    min_mse: float
) -> float:
    """
    Calculate the transformed MSE, i.e., 
    MSEtrans = (RMSE^2-RMSE_min-sigma)^2.
    This is used to find the uncertainty bounds.
    
    Parameters:
    -----------
        d: np.ndarray
            Array of distances.
        z1: np.ndarray
            First array of elevations with `z1.shape = d.shape`.
        z2: np.ndarray
            Second array of elevations with `z2.shape = z1.shape`
        sigma: float
            Uncertainty cutoff criterion after Wei et al. (2015).
        min_mse: float
            MSE of the best fit profile.
    
    Results:
    --------
        trans_mse: float
            Transformed MSE of the input elevation arrays.
    """

    D = d[-1]-d[0]
    f_i = np.zeros(len(z1))
    f_i[1:] = (d[1:] - d[:-1]) / D 
    f_i *= (len(f_i)/np.sum(f_i))
    
    trans_mse = (np.sum(f_i*((z1-z2)**2))/len(z1)-min_mse-sigma)**2
    
    return trans_mse 

def _nonlin_invert_uncertainty(
    d: np.ndarray, 
    z: np.ndarray, 
    nonlin_d: np.ndarray,
    t_best: float, 
    sigma: float, 
    min_mse: float, 
    S_c: float, 
    dt: float = 0.5, 
    max_iteration: int = 10_000,
    geom_params: dict = {}
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
    n = int(t_best / dt) + 1
    
    z_nl, t_nl = nonlin_diff_perron2011(
        z_init=z_init,
        dx=nonlin_d[1]-nonlin_d[0],
        dt=dt,
        n_t=n,
        S_c=S_c,
        n=2,
        k=1
    )
    
    # starting from the last step in t_nl,
    # continuously calculate the transformed mse
    # until MSE at new step is larger than the last one.
    # (i.e. it must be after the local minimum in the
    # transformed mse)
    
    intfun = sp.interpolate.interp1d(nonlin_d, z_nl[-1,:], kind="cubic")
    z_last = intfun(d)
    mse_reference = _transform_mse(d, z, z_last, sigma, min_mse)
    iteration = 0
    mse_out = -np.infty

    while (iteration < max_iteration) and (mse_out < 2*mse_reference):
        
        # calculate another step with nonlinear diffusion and add to
        # z_nl, t_nl...
        z_one, _ = nonlin_diff_perron2011(
            z_nl[-1,:], dx=nonlin_d[1]-nonlin_d[0], dt=dt, n_t=1, k=1, S_c=S_c,
            n=2
        )

        z_nl = np.concatenate((z_nl, [z_one[-1,:]]), axis=0)
        t_nl = np.concatenate((t_nl, [t_nl[-1]+dt]), axis=0)
        
        # calculate the new mse...
        intfun = sp.interpolate.interp1d(nonlin_d, z_nl[-1,:], kind="cubic")
        z_new = intfun(d)
        mse_out = _transform_mse(d, z, z_new, sigma, min_mse)
    
    # also look into the lower direction: what is the first point below t_best
    # at which the mse is higher than before
    
    ids = np.where(t_nl<t_best)[0][::-1]
    i = 0
    t_lower_bound = 0. 

    found_lower_bound = False
    while not found_lower_bound:
        
        # if we have iterated over all ids, set lb to 0.
        if len(ids) == i:
            t_lower_bound = 0.
            break
        
        intfun = sp.interpolate.interp1d(
            nonlin_d, 
            z_nl[ids[i],:], 
            kind="cubic"
        )
        z_i = intfun(d)
        mse_out = _transform_mse(d, z, z_i, sigma, min_mse)
        
        if mse_out > 2*mse_reference:
            t_lower_bound = t_nl[ids[i]]
            found_lower_bound = True

        i += 1
    
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
        x0=np.array([(t_lower_bound+t_best)/2]),
        args=args,
        method="Powell",
        bounds=((t_lower_bound, t_best),)
    )
    
    # second part
    id_upper = np.where(t_nl >= t_best-dt)[0]

    args = (
        z_nl[id_upper,:], t_nl[id_upper], 
        nonlin_d, S_c, d, z, sigma, min_mse
    )
    
    upper_opt = sp.optimize.minimize(
        fun=_nonlin_transform_mse_at_t,
        x0=np.array([(t_best+t_nl[-1])/2]),
        args=args,
        method="Powell",
        bounds=((t_best, t_nl[-1]),)
    )

    return lower_opt.x[0], upper_opt.x[0]