from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Union, Tuple
from typing_extensions import Self # pre python 3.11

import sys, os

import numpy as np
import warnings
from scipy.linalg import solve_banded
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

# small numeric checker
def _is_numeric(value):
    try:
        a = float(value)
        return True
    except:
        return False
    
def lin_diffusion_exp_fwd_time(
    z_init: np.array,
    dx: float,
    dt: float,
    n_t: int,
    k: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    lin_diffusion_exp_fwd_time: Function to calculate the evolution of a
    one-dimensional surface morphology according to the linear diffusion
    equation dz/dt = k * d^2z / dx^2. The explicit solution is calculated
    forwards in time and centered in space. Boundary conditions should be given
    as elevation values in z_init[0] and z_init[-1].

    Parameters:
    -----------
        z_init: np.ndarray
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float
            Diffusivity constant in m^2 / kyr.

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """

    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init

    t_steps = np.zeros(n_t+1)

    # copy boundaries from intial profile
    prof_matrix[:,0] = z_init[0]
    prof_matrix[:,-1] = z_init[-1]

    # define constant:

    C = (k * dt) / (dx**2)

    # check for stability:

    if C >= 0.5:
        warnings.warn(
            "Courant–Friedrichs–Lewy condition not met"
        )

    for i in range(1, n_t+1):
        prof_matrix[i,1:-1] = prof_matrix[i-1,1:-1] + \
            C*(prof_matrix[i-1,2:]-2* prof_matrix[i-1,1:-1]+prof_matrix[i-1,:-2])
        t_steps[i] = t_steps[i-1] + dt

    return (prof_matrix, t_steps)


class TridiagonalElimination:  # to be used in lin_diffusion_impl_ ...
    """
    Solve matrix elimination problem of form M*y = r:

    | b[0] c[1]                              || y[0] |   | r[0] |
    | a[0] b[1] c[2]                         || y[1] |   | r[1] |
    |      a[1] b[2] c[3]                    || y[2] |   | r[2] |
    |           .    .    .                  || .    | = | .    |
    |                .    .    .             || .    |   | .    |
    |                     .    .      c[n]   || .    |   | .    |
    |                          a[n-1] b[n]   || y[n] |   | r[n] |

    for y[:].
    """

    def __init__(
        self, 
        length: int
    ) -> None:
        """
        Initialize a TridiagonalElimination instace.
        
        Parameters:
        ----------- 
            length: int
                The size of the diagonals. The main diagonal has 
                length 0 to n.
        
        Returns:
        --------
            None
        """
        self.a = np.zeros(length)
        self.b = np.ones(length)
        self.c = np.zeros(length)
        self.r = np.zeros(length)

    def solve(self) -> np.ndarray:
        """
        Solve the matrix equation for y.
        
        Parameters:
        -----------
            None
        
        Returns:
        --------
            y: np.ndarray
                The solution to M*y = r
        """
        lhs = np.vstack(( self.c, self.b, self.a ))
        y = solve_banded((1, 1), lhs, self.r)
        return y



def lin_diffusion_impl_backwd_time(
    z_init: np.ndarray,
    dx: float,
    dt: float,
    n_t: int,
    k: float
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to calculate the evolution of a one-dimensional surface 
    morphology according to the linear diffusion equation 
    dz/dt = k * d^2z / dx^2. The implicit solution is calculated
    backwards in time and centered in space. Boundary conditions should be given
    as elevation values in z_init[0] and z_init[-1].

    Parameters:
    -----------
        z_init: np.ndarray
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float
            Diffusivity constant in m^2 / kyr.

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """

    solver = TridiagonalElimination(len(z_init))
    C = (dt * k) / (dx ** 2)
    solver.a[:-2] = -C
    solver.b[1:-1] = 1 + 2*C
    solver.c[2:] = -C
    solver.r = z_init.copy()

    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init

    times = np.zeros(n_t+1)

    for i in range(1, n_t+1):
        prof_matrix[i,:] = solver.solve()
        solver.r = prof_matrix[i,:]
        times[i] = times[i-1] + dt

    return (prof_matrix, times)


def lin_diffusion_crank_nicolson(
    z_init: np.ndarray,
    dx: float,
    dt: float,
    n_t: int,
    k: Union[float, ArrayLike]
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to calculate the evolution of a
    one-dimensional surface morphology according to the linear diffusion
    equation dz/dt = k * d^2z / dx^2. This function is based on the
    Crank-Nicolson method for numerically solving differential equations and
    indends to archieve a compromise between stability and accuracy.

    Parameters:
    -----------
        z_init: np.ndarray
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float
            Diffusivity constant in m^2 / kyr.

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """
    # if k is float, build array of length n_t
    try:
        k = float(k)
        k = np.full(n_t+1, k)
    except TypeError:
        if len(k) != n_t+1:
            raise Exception("len(k) != n_t")

    # create a C vector of length n_t
    C = [(dt*k_i)/(2*dx**2) for k_i in k]

    solver = TridiagonalElimination(len(z_init))

    solver.r = z_init.copy()  # for emplacing boundary conditions

    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init
    times = np.zeros(n_t+1)

    # initial RHS vector for Crank-Nicolson
    solver.r[1:-1] = C[0]*z_init[:-2] + \
        (1-2*C[0])*z_init[1:-1] + C[0]*z_init[2:]

    for i in range(1, n_t+1):
        #C = (dt * k[i-1]) / (2 * dx ** 2)
        solver.a[:-2] = -C[i]
        solver.b[1:-1] = 1 + 2*C[i]
        solver.c[2:] = -C[i]

        prof_matrix[i,:] = solver.solve()
        # update RHS
        solver.r[1:-1] = C[i]*prof_matrix[i,:-2] + \
            (1-2*C[i])*prof_matrix[i,1:-1] + C[i]*prof_matrix[i,2:]
        times[i] = times[i-1] + dt

    return (prof_matrix, times)

def nonlin_diffusion_explicit(
    z_init: np.ndarray,
    dx: float,
    dt: float,
    n_t: float,
    k: float,
    S_c: float,
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    nonlin_diffusion_explicit: Function to calculate the evolution of a
    one-dimensional surface morphology according to the nonlinear diffusion
    equation dz/dt = d/dx[f(x,k)dz/dx]. Where f(x,k) = k / (1-grad(z(x))/S_c).

    Parameters:
    -----------
        z_init: np.ndarray
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float
            Diffusivity constant in m^2 / kyr.
        S_c: float
            Critical slope.
        n: int
            Exponent in the nonlinear transport term.

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """

    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init

    # copy boundaries from intial profile
    prof_matrix[:,0] = z_init[0]
    prof_matrix[:,-1] = z_init[-1]

    t_steps = np.zeros(n_t+1)

    # stability: according to Perron 2011

    slope = (z_init[1:] - z_init[:-1]) / dx
    K_nl = k*(slope/S_c)**2 * (3-(slope/S_c)**2) / (1-(slope/S_c)**2)**2

    if dt >= (dx**2) / K_nl.max():
        dt_lim = (dx**2) / K_nl.max()
        warnings.warn(f"dt of {dt} may cause instability. Should be smaller than {dt_lim}.")


    # define constant:
    C = (k * dt) / (dx**2)

    for i in range(1, n_t+1):
        #with np.printoptions(threshold=np.inf):
            #print(prof_matrix[i-1,:])
        diff1 = prof_matrix[i-1,1:-1] - prof_matrix[i-1,:-2]
        grad1 = (diff1 / (dx*S_c)) ** n
        diff2 = prof_matrix[i-1,2:] - prof_matrix[i-1,1:-1]
        grad2 = (diff2 / (dx*S_c)) ** n
        prof_matrix[i,1:-1] = prof_matrix[i-1,1:-1] + \
            C * (((1 / (1-grad2)) * diff2 - (1 / (1-grad1)) * diff1))
        t_steps[i] = t_steps[i-1] + dt

    return (prof_matrix, t_steps)



def nonlin_diffusion_explicit_Perron2011(
    z_init: np.ndarray,
    dx: float,
    dt: float,
    n_t: float,
    k: float,
    S_c: float,
    uplift_rate: float = 0.,
    rho_ratio: float = 1.
) -> Tuple[np.ndarray, np.ndarray]:
    """
    nonlin_diffusion_explicit: Function to calculate the evolution of a
    one-dimensional surface morphology according to the nonlinear diffusion
    equation dz/dt = d/dx[f(x,k)dz/dx]. Where f(x,k) = k / (1-grad(z(x))/S_c).

    Parameters:
    -----------
        z_init: np.array
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float
            Diffusivity constant in m^2 / kyr.
        S_c: float
            Critical slope.
        uplift_rate: float
            Constant uplift rate applied to the profile.
        rho_rtio: float
            Ratio of bedrock to sediment density. One by default.

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """

    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init

    # copy boundaries from intial profile
    prof_matrix[:,0] = z_init[0]
    prof_matrix[:,-1] = z_init[-1]

    t_steps = np.zeros(n_t+1)

    # stability: according to Perron 2011

    slope = (z_init[1:] - z_init[:-1]) / dx
    K_nl = k*(slope/S_c)**2 * (3-(slope/S_c)**2) / (1-(slope/S_c)**2)**2

    if dt >= (dx**2) / K_nl.max():
        dt_lim = (dx**2) / K_nl.max()
        warnings.warn(f"dt of {dt} may cause instability. Should be smaller than {dt_lim}.")

    for i in range(1, n_t+1):
        
        # this computation follows Perron (2011)
        
        # compute zx and zxx at t_n:
        z_x = (prof_matrix[i-1,2:] - prof_matrix[i-1,:-2]) / (2*dx)
        z_xx = (prof_matrix[i-1,2:] - \
            2*prof_matrix[i-1,1:-1] + prof_matrix[i-1,:-2]) / (dx**2)
        
        term1 = (z_xx) / (1 - (z_x / S_c)**2)
        term2 = (2*(z_x**2)*z_xx) / ((S_c**2)*(1-(z_x/S_c)**2))
        
        prof_matrix[i,1:-1] = prof_matrix[i-1,1:-1] + rho_ratio*uplift_rate*dt + \
            dt*k*(term1 + term2)
        t_steps[i] = t_steps[i-1] + dt

    return (prof_matrix, t_steps)

def nonlin_diff_perron2011(
    z_init: np.ndarray,
    dx: float,
    dt: float,
    n_t: float,
    k: Union[float, np.ndarray],
    S_c: float,
    n: float = 2.,
    warning_eps: float = None,
    uplift_rate: Union[np.ndarray, float] = 0.,
    rho_ratio: float = 1.
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the evolution of a
    one-dimensional surface morphology according to the nonlinear diffusion
    equation dz/dt = d/dx[f(x,k)dz/dx]. Where f(x,k) = k / (1-grad(z(x))/S_c)
    using the Q-imp scheme proposed by 
    `Perron 2011 <https://doi.org/10.1029/2010JF001801>`_.

    Parameters:
    -----------
        z_init: np.ndarray
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float | np.array
            Diffusivity constant in m^2 / kyr. If array, length
            should equal the number of time steps taken.
        S_c: float
            Critical slope.
        n: int
            Exponent in the nonlinear transport term.
        warning_eps: float
            Print a warning message if slopes are higher than this value. Defaults to
            the value of S_c.
        uplift_rate: np.ndarray | float
            Imposed uplift rate in m/kyr. Zero by default. If supplied as array,
            len(uplift rate) == n_t must be True. No uplift is applied
            at the boundaries.
        rho_ratio: float
            Ratio of rock density to sediment density. (Assuming that
            the supplied uplift rate is a rock uplift rate, and rock
            decreases in density while it is supplied to the landscape.)

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """

    # convert uplift_rate to array, if needed:
    if _is_numeric(uplift_rate):
        uplift_rate_time_array = np.full(n_t, uplift_rate)
    else:
        if len(uplift_rate) != n_t:
            raise Exception("len(uplift_rate) != n_t")
        uplift_rate_time_array = uplift_rate 

    # convert k to array, if needed
    if _is_numeric(k):
        k_array = np.full(n_t, k)
    else:
        if len(k) != n_t:
            raise Exception("len(k) != n_t")
        k_array = k
    
    if warning_eps == None: warning_eps = S_c
    
    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init

    # copy boundaries from intial profile
    prof_matrix[:,0] = z_init[0]
    prof_matrix[:,-1] = z_init[-1]

    t_steps = np.zeros(n_t+1)

    solver = TridiagonalElimination(len(z_init))
    # place boundary conditions:
    solver.r = z_init.copy()

    uplift_rate_space_array = np.zeros(z_init.shape)
    
    WARNING_FLAG = True 
    
    for i in range(1, n_t+1):

        k = k_array[i-1] # always of current time step...
        
        # no uplift at boundaries
        uplift_rate_space_array[1:-1] = uplift_rate_time_array[i-1]

        # second and first derivative
        z_x = (prof_matrix[i-1,2:]-prof_matrix[i-1,:-2])/(2*dx)
        z_xx = \
            (prof_matrix[i-1,2:]-2*prof_matrix[i-1,1:-1]+prof_matrix[i-1,:-2])/\
            (dx**2)

        # if the original z_x has negative slopes, then those slopes are
        # permitted. (e.g. for b<0) Set warning_eps to np.nan
        if any(np.abs(z_x)>warning_eps) and WARNING_FLAG: 
            WARNING_FLAG = False
            warnings.warn(f"z_x=abs({z_x.min():.4f}) > {warning_eps:.4f}: Can only be caused by instability.")

        # a = k, for comparability with Perron 2011.
        b = 1 / (S_c**n)  # introduced by Perron 2011.
        c = 1 / (1 - b * z_x ** n)  # introduced by Perron 2011.

        F_i = (-2*k*c/(dx**2))*(1+n*b*c*z_x**n)  # F_i^n
        
        term1 = k*c/(dx**2)*(1+n*b*c*z_x**n)
        term2 = ((n**2+n)/2)*((k*b*c**2*z_x**(n-1)*z_xx)/(dx))
        term3 = (n**2*k*b**2*c**3*z_x**(2*n-1)*z_xx)/dx
        
        F_ip1 = term1 + term2 + term3  # F_{i+1}^n
        F_im1 = term1 - term2 - term3  # F_{i-1}^n

        # f(z_i^n): this is the change in elevation at time step n,
        # as stated by eq. (6) in Perron (2011)
        fzin = k*(z_xx*c + n*b*c**2*z_x**n*z_xx) + \
            rho_ratio*uplift_rate_space_array[1:-1]

        # build tridiagonal matrix
        solver.a[:-2] = -dt*F_im1
        solver.b[1:-1] = 1-dt*F_i
        solver.c[2:] = -dt*F_ip1
        solver.r[1:-1] = prof_matrix[i-1,1:-1] + \
            dt*(fzin-F_im1*prof_matrix[i-1,:-2] - \
            F_i*prof_matrix[i-1,1:-1] - \
            F_ip1*prof_matrix[i-1,2:])

        prof_matrix[i,:]= solver.solve()

        t_steps[i] = t_steps[i-1] + dt

        #print(prof_matrix[i,:])
    return (prof_matrix, t_steps)

def nonlin_diff_gabet2021(
    z_init: np.array,
    dx: float,
    dt: float,
    n_t: float,
    k: float,
    warning_eps: float = -10e-15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the evolution of a
    one-dimensional surface morphology according to the nonlinear diffusion
    equation dz/dt = d/dx[f(x,k)dz/dx]. Where f(x,k) = k * (dz/dx)**2, as 
    proposed by `Gabet et al. 2021 <https://doi.org/10.1029/2020JF005858>`_ 
    using the Q-imp scheme of Perron 2011.

    Parameters:
    -----------
        z_init: np.ndarray
            Initial elevation profile, assumed to be evenly spaced with
            distances dx and boundary elevations that will not change at
            z_init[0] and z_init[-1].
        dx: float
            Spacing between adjacent elevation values in meters,
            assumed to be constant.
        dt: float
            Size of the time step taken, assumed to be constant. In kilo-years.
        n_t: int
            Number of timesteps. The oldest age will consequently be dt*n_t.
        k: float
            Diffusivity constant in m^2 / kyr.
        warning_eps: float
            Print a warning if slopes get more negative than this value.

    Returns:
    --------
        prof_matrix: np.ndarray
            Matrix containing rows of profiles, first row being the initial
            profile, last row being the profile at time dt*n_t.
        t_steps: np.ndarray
            1D array containing time steps at which the profile was calculated.

    """

    prof_matrix = np.zeros((n_t+1, len(z_init)))
    prof_matrix[0,:] = z_init

    # copy boundaries from intial profile
    prof_matrix[:,0] = z_init[0]
    prof_matrix[:,-1] = z_init[-1]

    t_steps = np.zeros(n_t+1)

    solver = TridiagonalElimination(len(z_init))
    # place boundary conditions:
    solver.r = z_init.copy()

    WARNING_FLAG = True
    for i in range(1, n_t+1):

        z_x = (prof_matrix[i-1,2:]-prof_matrix[i-1,:-2])/(2*dx)
        z_xx = \
            (prof_matrix[i-1,2:]-2*prof_matrix[i-1,1:-1]+prof_matrix[i-1,:-2])/\
            (dx**2)

        if any(z_x<warning_eps) and WARNING_FLAG: # floating point stuff...
            WARNING_FLAG = False
            warnings.warn(f"z_x={z_x.min():.4f} < 0: Can only be caused by instability.")

        F_i = (-4*k*z_x) / (dx**2)
        F_ip1 = (2*k*z_x)/(dx**2) + (k*z_xx)/dx
        F_im1 = F_ip1
        # f(z_i^n) erosion rate:

        erosion_rate = 2*k*z_x*z_xx

        # build tridiagonal matrix
        solver.a[:-2] = -dt*F_im1
        solver.b[1:-1] = 1-dt*F_i
        solver.c[2:] = -dt*F_ip1
        solver.r[1:-1] = prof_matrix[i-1,1:-1] + \
            dt*(erosion_rate-F_im1*prof_matrix[i-1,:-2] - \
            F_i*prof_matrix[i-1,1:-1] - \
            F_ip1*prof_matrix[i-1,2:])

        prof_matrix[i,:]= solver.solve()

        t_steps[i] = t_steps[i-1] + dt

        #print(prof_matrix[i,:])
    return (prof_matrix, t_steps)

def animate_profile_diffusion(
    z_nonlin_matrix: np.ndarray,
    t_steps: np.ndarray,
    dx: float = 1.,
    d_nonlin: np.ndarray = np.array([]),
    d_opt: np.ndarray = np.array([]),
    z_opt: np.ndarray = np.array([]),
    name: str = "Profile",
    outputgif: bool = False,
    outputname: str = ""
) -> None:
    """
    Create a GIF based on a precomputed profile diffusion evolution.
    
    Parameters:
    -----------
        z_nonlin_matrix: np.ndarray
            Profile at different time steps.
        t_steps: np.ndarray
            The time steps.
        dx: float
            Spacing between computed grid points. Used if no d_nonlin
            is supplied.
        d_nonlin: np.ndarray
            Spacing along the calculated profile.
        d_opt: np.ndarray
            Optional, the original recorded profile elevations.
        z_op: np.ndarray
            Optional, the original recorded profile distances.
        name: str
            Profile name.
        outputgif: bool
            Whether to output the animation as a GIF.
        outputname: str
            The subdirectory and name of the file.
        
    Returns:
    --------
        None
    """

    def animate(i):
        title = f"t = {t_steps[i]:.2f} kyr"
        ax.clear()
        ax.plot(d_opt, z_opt, c="red")
        ax.plot(d_nonlin, z_nonlin_matrix[i,:], linestyle="solid", c="black")

        # ax.legend(frameon=False, title="Diffusion")
        ax.set_title(title, loc="left")
        ax.set_title(name)
        ax.set_ylabel("Relative elevation [m]")
        ax.set_xlabel("Relative distance [m]")
        #ax.set_ylim(z_nl[0,:].min

    # distances:
    if d_nonlin.shape == (0,):
        # construct d using dx
        n_d = z_nonlin_matrix.shape[1] # number of d points needed

        d_nonlin = np.arange(0, (n_d)*dx, dx)
        d_nonlin = d_nonlin - d_nonlin.max()/2


    fg, ax = plt.subplots()

    ani = FuncAnimation(fg, animate, frames=len(t_steps), interval=1e-6,
        repeat=False)
    plt.show()

    writergif = PillowWriter(fps=30)
    if outputgif is True:
        ani.save(os.getcwd()+"//"+outputname+"diffusion_animation.gif",
                 writer=writergif)
