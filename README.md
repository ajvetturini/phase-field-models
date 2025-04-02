# phase-field-models
Python / JAX implementation of Cahn Hilliard from https://github.com/lorenzo-rovigatti/cahn-hilliard alongside reproduced results. The full details of this implementation can be found in their arXiv paper: https://arxiv.org/pdf/2501.04790. Overall, if you are unfamiliar with phase fields / other mesoscopic simulations, I'd recommend reading Ch 12 and 13 of Introduction to Computational Materials Science by Richard Lesar.

# Implementation Details

- The code here supports 1D and 2D simulations and take a TOML-file based input detailing the simulation parameters.
- Currently, landau, simple_wertheim, and saleh free energy models are implemented
