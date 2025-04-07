"""
Base class for Integrator modules
"""
import numpy as np
from typing import Optional

class Integrator:
    def __init__(self, model, config):
        self._model = model
        self._N_per_dim = config.get('N', 64)
        self._dt = config.get('dt', 0.001)
        self._k_laplacian = config.get('k', 1.0)
        self._M = config.get('M', 1.0)
        self._gamma = config.get('gamma', 1.0)
        self._dx = config.get('dx', 1.0)
        self._dim = config.get('dim', 2)
        if self._dim <= 0 or self._dim > 2:
            raise Exception('Unable to proceed, currently only support for 1D and 2D is implemented')

        num_species = self._model.N_species()
        shape = tuple([num_species] + [self._N_per_dim] * self._dim)  # we need spatial for each of the num_species!\
        self._rho = np.zeros(shape)
        self._N_bins = np.prod(self._rho.shape[1:])  # Total # of spatial bins (elements)

        # Setup scaling factor:
        distance_scaling_factor = config.get('distance_scaling_factor', 1.0)
        inverse_scaling_factor = 1.0 / distance_scaling_factor
        self._dx *= inverse_scaling_factor                      # Proportional to m
        self._M /= inverse_scaling_factor                       # Proportional to m^-1
        self._k_laplacian *= np.pow(inverse_scaling_factor, 5)  # Proportional to m^5

        self._inverse_scaling_factor = inverse_scaling_factor
        self._distance_scaling_factor = distance_scaling_factor
        self._use_autodiff = config.get('use_autodiff', False)  # Use manual derivative by default
        self._interface_scalar = config.get('interface_scalar', 1.)  # Scales interface energy

    def set_initial_rho(self, r):
        self._rho[:] = r[:]

    def evolve(self, rho: Optional):
        raise NotImplementedError("evolve must be implemented by derived classes.")

    def rho(self):
        return self._rho
