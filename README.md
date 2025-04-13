# Phase Field Modelling using JAX
Python / JAX implementation of Cahn-Hilliard and Allen-Cahn phase field model algorithms (inspired from https://github.com/lorenzo-rovigatti/cahn-hilliard) alongside reproduced results. The output of this phase field package (i.e., the resultant mass / energy log and order parameter trajectory) are formatted similarly to the cahn-hilliard package, and here we also implement Allen-Cahn model. Furthermore, because the performance is quite similar to the CUDA-based implementation (see below [Performance Comparison](#performance-comparison)), I hope this package may enable more rapid prototyping of free energy models. JAX also offers automatic differentiation, enabling access to the gradient updates during the energy minimization procedure which may be more interesting from a meta-optimization perspective. Overall, if you are unfamiliar with phase fields / other mesoscopic simulations, I'd recommend reading Ch 12 and 13 of Introduction to Computational Materials Science by Richard Lesar.

Overall, this library has a larger foucs on grid-based phase field modelling for self-assembly based systems, whereas other phase field model implementations (e.g., such as [JAX-AM](https://github.com/tianjuxue/jax-am)) focus on looking at grain development for additive manufacturing where the mesh quality is of greater importance.

Finally, these are lengthy simulations if you do not have a hardware accelerator (i.e., GPU / TPU). This jax-based version should still run on a CPU and allow you to still simulate more coarse / lower time step systems due to JIT with performance similar to a CPU-based C++ implementation. Overall, jax should be accessible on any type of computer (Windows, Mac, Linux) but Linux is preferred for easiest access to GPU acceleration. For more information, see [here](https://docs.jax.dev/en/latest/).

# Implementation Details

- The code here supports 1D and 2D simulations and take a TOML-file based input detailing the simulation parameters.
- Phase field models implemented: ``Cahn-Hilliard`` | ``Allen-Cahn`` |
- Free energy models implemented: ``Landau`` | 
- Numerical Integrators implemented: ``Explicit Euler`` |
- Only **periodic** boundary conditions are handled in this implementation. 
- Currently, all grids are handled with constant size $N$ and grid-cell size of $dx$
- Free energies are handled as *dimensionless*, and the equation constants (e.g., M / mobility constant in Cahn-Hilliard) is actually $M'=K_bTM$ which is treated as 1 (nm s $)^{-1}$
- The input files in the Examples directory will show how scaling constants (for the interface / bulk energy values in the Allen-Cahn or Cahn-Hilliard) can be incorporated through the input config TOML to better fit this package to your specific needs

# Tips
- Be careful with how the density field is handled (i.e., it's shape is one of: (Ns, Nx), (Ns, Nx, Ny) or (Ns, Nx, Ny, Nz) where Ns is the number of species in the simulation)!
- The [MagneticFilm.py](https://github.com/ajvetturini/phase-field-models/blob/main/Examples/MagneticFilm/run.py) example shows how you can incorporate your own energy model to be solved via Allen-Cahn or Cahn-Hilliard as well as specifying unique initialization conditions outside of the default method using the ``initial_density`` config parameter.
- If your system has multiple GPUs, you can run the following command in terminal to specify a device: ``` CUDA_VISIBLE_DEVICES="DEVICE_NUMBER" python run.py ``` where the run.py script will read in a TOML and start a run (see [this example](https://github.com/ajvetturini/phase-field-models/blob/main/Examples/Landau/cuda_vs_jax/run.py))
- When developing your own energy model, start with float32 precision. Increasing this to float64 is necessary for more complex free energy models (e.g., any of the Wertheim models). 
  - Float64 will greatly reduce the efficiency of the model (and is also dependent on GPU hardware)
  - The LaPlacian operator is the computational overhead since it is calculated twice during Cahn-Hilliard updates
  - Other jax-based implementations for handling the periodic boundary condition laplacian (e.g., FFT or Convolutions) did not seem to improve performance, but that may be due to my specific implementation!


# Future Features
Below is a list of features that would be nice to be implemented. Please let me know (see end of README / open a PR) if you want to see any other features added!

- simple_wertheim, and saleh free energy functions
- Analysis toolkit features
- Different numerical integrators
- Investigating periodic conditions in JAX

# Citations / Links
To learn more about the free energy models implemented in this package, please see the citations below. Also, if you used this package then let me know and I can add a citation here!

1) ``Wertheim Theory`` | Cappa, M., Sciortino, F., and Rovigatti, L., "A phase-field model for DNA self-assembly", arXiv (2025). | <a href="https://arxiv.org/pdf/2501.04790" target="_blank">LINK</a>
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
4) Install required dependencies (this is usually done automatically during step 3, but if you want to install JAX for GPU, then follow the first line below)
```
# For CUDA-enabled systems:
pip install "jax[cuda]" matplotlib toml tqdm

# For CPU-only or non-Linux systems (e.g., Windows/Mac):
pip install jax matplotlib toml tqdm
```

# Got questions?
Feel free to reach out to me at avetturi [at] andrew [dot] cmu [dot] edu or open a PR
