Tutorials
=========

Tutorials are provided as Jupyter Notebooks on the package 
`GitHub page <https://github.com/lengri/riserfit/tree/main/Tutorials>`_. If you have 
installed ``riserfit`` already, you can run::

    pip install riserfit[tutorials]

to install any additional dependencies needed for the tutorials.

The example data for the tutorials is a 1 m resolution DEM
in the ``\\Data\\DEM\\`` folder. It contains a small patch of terraces along the
Saxton River, New Zealand. The DEM was downloaded from `OpenTopography <https://opentopography.org>`_.
Original LiDAR dataset courtesy of Marlborough District Council and 
Toitū Te Whenua Land Information New Zealand (2023). 

Notebooks
---------

At the moment, there are five tutorial notebooks. 

**Notebook 1** deals with extracting data from a DEM to feed into
riserfit. It introduces risefits central class, ``riserfit.Riser``,
which handles data and most processing steps.

**Notebook 2** The second notebook deals with calculating diffusion ages,
the core business of morphological dating. This notebook is based on 
the *linear diffusion equation*.

**Notebook 3** This notebook extends morphological dating to the *nonlinear diffusion
equation*, which is not often used for morphological dating. Contrary to 
the implementation of `Xu et al. 2021 <https://doi.org/10.1002/esp.5022>`_, 
``riserfit`` implements a fully nonlinear sediment transport law.

**Notebook 4** The natural next step in morphological dating is to convert diffusion ages
[m:math:`^2`] into actual ages and accompanying diffusivity estimates (:math:`k`).
The fourth notebook gives an outline of this process.

**Notebook 5** The last notebook is quite different form the previous ones. It introduces the 
``riserfit.RiserPlayground`` class, which is used to generate synthetic profiles
with different magnitudes of (Gaussian) noise. This toolkit can be used to 
determine how inferred diffusion ages are affected by noise and how errors propagate.


