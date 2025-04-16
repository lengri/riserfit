Morphological dating 
====================

This page gives a brief overview of the mathematics and physics behind morphological dating.
The idea behind all forms of morphological dating is simple: Fault scarps, shorelines, 
marine terraces, and fluvial terrace risers are *transient* landscape features whose shape
changes continuously from the time of their creation until they are no longer recognizable
as scarps or risers. Thus, the degree of degradation can tell us something about the age of
the landform.

Risers and scarps have a steep face embedded in a sub-horizontal far field. Erosion acts to 
degrade steep slopes and will ultimately level the landform by removing material from the 
crest of the riser and depositing it at its base/toe. Because risers and scarps are smooth 
landforms, no convergent flow occurs on the landform itself. Erosion and flow on scarps and 
risers is diffuse in nature.

Hillslope diffusion
-------------------
Traditionally, diffusion equations are used to understand the evolution of risers and scarps.
Based on the assumption that sediment is eroded by rainsplash, freeze-thaw cycles, animal
burrowing, and plant uprooting, individual grains of sediment are assumed to travel only
small distances such that local sediment flux depends only on local slope conditions.
In other words: 

    .. math:: q_s = k\frac{\partial z}{\partial x}

Sediment that is added to the sediment flux :math:`q_s` has to come from somewhere, and
sediment removed from the flux has to go somewhere. This realisation takes the form of a
mass preservation statement on a *local* scale

    .. math:: \frac{\partial q_s}{\partial x} = \frac{\partial z}{\partial t}

By combining the linear flux-slope equation with the mass preservation statement, we arrive
at the linear diffusion equation

    .. math:: \frac{\partial z}{\partial t} = k\frac{\partial^2 z}{\partial x^2}.

Risers and scarps evolving according to this equation is a fundamental assumption of almost
all morphological dating efforts. Because we can quantify how fast a landform erodes 
(:math:`\partial z / \partial t`), we can calculate how long it must have taken for a scarp or
riser encountered in the field to reach its observed form.

Nonlinear Diffusion
-------------------

There is an active debate on whether linear diffusion accurately depicts erosion on steep slopes.
More specifically, linear diffusion predicts a convex-up hillslope form, while many hillslopes in 
nature have near zero concavitiy at steep slopes 
(`Roering et al. 1999 <https://doi.org/10.1029/1998WR900090>`_). This led 
`Andrews and Bucknam (1987) <https://doi.org/10.1029/JB092iB12p12857>`_ and 
`Roering et al. (1999) <https://doi.org/10.1029/1998WR900090>`_ to propose a nonlinear 
dependence of sediment flux on slope, approaching infinity for slopes close to some 
critical slope :math:`S_c`:

    .. math:: q_s = \frac{k\frac{\partial z}{\partial x}}{1-\left(S_c^{-1}\frac{\partial z}{\partial x}\right)^2}

which leads to an alternative, nonlinear diffusion equation

    .. math:: \frac{\partial z}{\partial t} = \frac{\partial}{\partial x}\left(\frac{k\frac{\partial z}{\partial x}}{1-\left(S_c^{-1}\frac{\partial z}{\partial x}\right)^2}\right)

Both equations, linear and nonlinear diffusion, have been used to infer morphological ages of scarp-like 
features, and both equations can be used in riserfit.

Nonlocal models
---------------

Recently, nonlocal models of sediment transport have become more popular to analyse the evolution 
of fault scarps specifically (e.g. `Gray et al. 2025 <https://doi.org/10.1130/G52987.1>`_). 
At the moment, nonlocal methods are not implemented in riserfit.