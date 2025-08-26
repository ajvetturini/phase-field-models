"""
Various constants used throughout the simulation code
"""
from typing import List

class Species:
    """
    Represents a chemical species with its patches, used in GenericWertheim.
    Example: patches=[0,0,1,1] -> unique_patches=[(0,2),(1,2)], N_unique_patches=2
    """
    def __init__(self, idx: int, patches: List[int]):
        self.idx = idx
        self.patches = patches

        # compute unique patches + multiplicities
        counter = {}
        for val in patches:
            counter[val] = counter.get(val, 0) + 1
        self.unique_patches = [UniquePatch(k, v) for k, v in counter.items()]
        self.n_unique_patches = len(self.unique_patches)

class UniquePatch:
    def __init__(self, idx: int, multiplicity: int):
        self.idx = idx
        self.multiplicity = multiplicity

class PatchInteraction:
    def __init__(self, species: int = -1, patches: List[UniquePatch] = None):
        self.species = species
        self.patches = patches if patches is not None else []


# GLOBAL CONSTANTS:
kb = 1.9872036  # i.e., ideal gas constant, used in Delta

