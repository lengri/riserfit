import sys, os
testdir = os.path.dirname(__file__)
srcdir = '../src/'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
os.chdir("C:/Users/Lennart/lennartGit/riserfitTesting")

import riserfit as rfc
import numpy as np
import matplotlib.pyplot as plt

# initial profile:
dx = 1
kt = 0  # assume k = 1, t = 0.
d = np.arange(-300, 300, dx)
a = 10
b = 0.01
theta = 0.4
z0 = rfc.analytical_profile(d=d,
    kt = kt,
    a = a,
    b = b,
    theta = theta
    )

k = 1
n = 2.
S_c = 0.5
dt = 1
n_t = 1000

z_cn, t_cn = rfc.lin_diffusion_crank_nicolson(
    z0,
    dx,
    dt=dt,
    n_t=n_t,
    k=k
    )

step_mult = [1, 5, 10, 100]
errors = [] # will be nested list
times = [] # also a nested list

z_nl, t_nl = rfc.nonlin_diff_perron2011(
    z0,
    dx,
    dt = dt,
    n_t = n_t,
    k=k,
    S_c=S_c,
    n=2
    )

rfc.animate_profile_diffusion(z_nl, t_nl, d_nonlin=d,
                              z_opt = z_nl[0,:], d_opt=d)

for i, step in enumerate(step_mult):

    z_nl_step = z_nl[::step,:]
    t_nl_step = t_nl[::step]

    z_nl_imp, t_nl_imp =rfc.nonlin_diff_perron2011(
        z0,
        dx,
        dt=dt * step,
        n_t= int(n_t / step),
        k=k,
        S_c=S_c,
        n=2,
        method="v2"
    )

    if i == 0:
        z_gif = z_nl_imp
        t_gif = t_nl_imp

    # calculate error between z_nl and z_nl_imp:
    error = np.sum(np.sqrt(((z_nl_step-z_nl_imp)**2) ), axis=1) / z_nl_step.shape[1]
    errors.append(error[1:])
    times.append(t_nl_step)
    
# plot error against time for each step size
lstyle = ["-", "--", "-.", ":"]
cGray=[0.2,0.2,0.2]
fg, ax = plt.subplots(1,2, figsize=(10,5))

for i, error in enumerate(errors):
    ax[0].plot(times[i][1:], error, linestyle=lstyle[i],
             c=cGray, label=f"{step_mult[i]}")
ax[0].legend(title="Implicit dt factor", frameon=False, ncol=2)
ax[0].set_xlabel("Time [kyr]")
ax[0].set_ylabel("RMSE [m]")
ax[0].ticklabel_format(scilimits=(0,10),
                     useMathText=True)

ax[1].plot(d, z_nl_step[2,:])
ax[1].plot(d, z_nl_imp[2,:])
plt.show()

