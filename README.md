# Phase Field Modelling using JAX
Python / JAX implementation of Cahn-Hilliard and Allen-Cahn phase field model algorithms (inspired from https://github.com/lorenzo-rovigatti/cahn-hilliard) alongside reproduced results. Overall, if you are unfamiliar with phase fields / other mesoscopic simulations, I'd recommend reading Ch 12 and 13 of Introduction to Computational Materials Science by Richard Lesar.

Overall, these are lengthy simulations if you do not have a hardware accelerator (i.e., GPU / TPU). This jax-based version should still run on a CPU and allow you to still simulate more coarse / lower time step systems due to JIT. Overall, jax should be accessible on any type of computer (Windows, Mac, Linux) but Linux is preferred for easiest access to GPU acceleration. For more information, see [here](https://docs.jax.dev/en/latest/)

# Implementation Details

- The code here supports 1D and 2D simulations and take a TOML-file based input detailing the simulation parameters.
- Free energy models suppoorted: ``Landau`` | 
- Numerical Integrators supported: ``Explicit Euler`` |
- Automatic Differentiation is supported for the free energy models. However, you must be careful about dimensionality and size. If N is large with many species, you're array will grow very large and memory reuirements will become a concern. 
  - If you are developing your own energy model for a process and want to differentiate through the updates, you will also need to determine if you need to use jax.jacobian based on your free energy model.
- Currently, all grids are handled with constant size $N$ and grid-cell size of $dx$
- Be careful with how the density field is handled (i.e., it's shape)!

# Future Features
Below is a list of features that would be nice to be implemented. Please let me know (see end of README / open a PR) if you want to see any other features added!

- simple_wertheim, and saleh free energy functions
- 3D support
- Analysis toolkit features
- Different numerical integrators

# Citations / Links
To learn more about the free energy models implemented in this package, please see the citations below. If you used this package and would like to have a paper listed here, please let me know.

1) ``Wertheim Theory`` | Capppa, M., Sciortino, F., Rovigatti, L., "A phase-field model for DNA self-assembly", arXiv (2025). | [LINK](https://arxiv.org/pdf/2501.04790)
2) ``Saleh Model`` | Jeon, B. Nguyen, D. T., and Saleh, A. O., "Sequence-controlled adhesion and microemulsification in a two-phase system of DNA liquid droplets", Journal of Physical Chemistry 123 (2020). | <a href="https://pubs.acs.org/doi/10.1021/acs.jpcb.0c06911" target="_blank">LINK</a>

# Performance Comparison
Coming soon!

# Package Requirements
This project requires Python 3.12. Due to the rapid development of JAX, it is strongly recommended to install dependencies in a clean Python 3.12 virtual environment. The following instructions assume you are using `conda`.

1) Create and activate a new environment
```
conda create -n [env_name] python=3.12
conda activate [env_name]
```
2) Clone the repository
```
git clone https://github.com/ajvetturini/phase-field-models
cd phase-field-models
```
3) Install the package in editable mode
```
pip install -e .
```
4) Install required dependencies
```
# For CUDA-enabled systems:
pip install "jax[cuda]" matplotlib toml

# For CPU-only or non-Linux systems (e.g., Windows/Mac):
pip install jax matplotlib toml
```


