# Phase Field Modelling using JAX
Python / JAX implementation of Cahn Hilliard from https://github.com/lorenzo-rovigatti/cahn-hilliard alongside reproduced results. The full details of this implementation can be found in their [arXiv paper](https://arxiv.org/pdf/2501.04790). Overall, if you are unfamiliar with phase fields / other mesoscopic simulations, I'd recommend reading Ch 12 and 13 of Introduction to Computational Materials Science by Richard Lesar.

Overall, these are lengthy simulations if you do not have a hardware accelerator (i.e., GPU / TPU). This jax-based version should still run on a CPU and allow you to still simulate more coarse / lower time step systems due to JIT. Overall, jax should be accessible on any type of computer (Windows, Mac, Linux) but Linux is preferred for GPU acceleration. For more information, see [here](https://docs.jax.dev/en/latest/)

# Implementation Details

- The code here supports 1D and 2D simulations and take a TOML-file based input detailing the simulation parameters.
- Free energy models suppoorted: ``Landau`` | 
- Numerical Integrators supported: ``Explicit Euler`` |
- Automatic Differentiation is supported for the free energy models. However, you must be careful about dimensionality and size. If N is large with many species, you're array will grow very large and memory reuirements will be come a concern. 
  - If you are developing your own energy model for a process and want to differentiate through the updates, you will also need to determine if you need to use jax.jacobian based on your free energy model.
- Currently, all grids are handled with constant size $N$ and grid-cell size of $dx$
- Be careful with how the density field is handled (i.e., it's shape). The 2D rho field is $(N_s, N_y, N_x)$ for example, which may be counterintuitive. 

# Future Features?
Below is a list of features that would be nice to be implemented. Open a PR if you want to suggest any.

- simple_wertheim, and saleh free energy functions
- 3D support
- Allen-Cahn model

# Want to help?
Reach out to me at avetturi [@] andrew [dot] cmu [dot] edu!