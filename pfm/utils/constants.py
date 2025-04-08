"""
Various constants used throughout the simulation code

I also through in the Species class here used in certain modules
"""
from typing import List, Dict, Tuple

class Species:
    """Represents a chemical species with its patches."""
    def __init__(self, idx: int, n_patches: int, patches: List[int]):
        self.idx = idx
        self.n_patches = n_patches
        self.patches = patches


# GLOBAL CONSTANTS:
kb = 1.9872036  # i.e., ideal gas constant, used in Delta

