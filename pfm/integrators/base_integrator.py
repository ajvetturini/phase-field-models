"""
Base class for Integrator modules
"""
import numpy as np

class Integrator:
    def __init__(self, model, config):
        self._model = model
        self._N_per_dim = config.get('N', 64)
        self._dt = config.get('dt', 0.001)
        self._k_laplacian = config.get('k_laplacian', 1.0)
        self._M = config.get('M', 1.0)
        self._dx = config.get('dx', 1.0)
        self._dim = config.get('dim', 2)
        if self._dim <= 0 or self._dim > 2:
            raise Exception('Unable to proceed, currently only support for 1D and 2D is implemented')
        self._rho = np.zeros(tuple([self._N_per_dim] * self._dim))
        self._N_bins = np.prod(self._rho.shape)
        self._user_to_internal = 1.0  # TODO: Replace these two values, what are these?
        self._internal_to_user = 1.0

    def set_initial_rho(self, r):
        self._rho[:] = r[:]

    def evolve(self):
        raise NotImplementedError("evolve must be implemented by derived classes.")

    def rho(self):
        return self._rho
