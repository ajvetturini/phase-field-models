import numpy as np
from typing import Optional
import jax
import jax.numpy as jnp

def make_laplacian_roll(dim, dx):
    def laplacian_single_species(phi):
        result = -2 * dim * phi  # center term

        if dim >= 1:
            result += jnp.roll(phi, shift=+1, axis=0) + jnp.roll(phi, shift=-1, axis=0)
        if dim >= 2:
            result += jnp.roll(phi, shift=+1, axis=1) + jnp.roll(phi, shift=-1, axis=1)
        if dim == 3:
            result += jnp.roll(phi, shift=+1, axis=2) + jnp.roll(phi, shift=-1, axis=2)

        return result / (dx**2)

    return jax.jit(jax.vmap(laplacian_single_species))

class Integrator:
    def __init__(self, model, config):
        self._model = model
        self._N_per_dim = config.get('N', 64)
        self._dt = config.get('dt', 0.001)
        self._k_laplacian = config.get('k', 1.0)
        self._M = config.get('M', 1.0)
        self._L_phi = config.get('L_phi', 1.0)
        self.dx = config.get('dx', 1.0)
        self._dim = config.get('dim', 2)
        self._float_type = config.get('float_type', jnp.float32)
        if isinstance(self._float_type, str):
            if self._float_type == 'float64':
                self._float_type = jnp.float64
            elif self._float_type == 'float32':
                self._float_type = jnp.float32
            else:
                raise Exception(f'Invalid float_type specified: {self._float_type} | Valid options are `float64` and '
                                f'`float_32`.')

        if self._dim <= 0 or self._dim > 3:
            raise Exception('Unable to proceed, package only supports periodic 1D - 3D')

        # Ensure square grid:
        assert self._N_per_dim > 0 and (self._N_per_dim & (self._N_per_dim - 1)) == 0, "N_per_dim must be a power of 2"
        num_species = self._model.N_species()
        shape = tuple([num_species] + [self._N_per_dim] * self._dim)  # we need spatial for each of the num_species!\
        self._rho = np.zeros(shape)
        self._N_bins = np.prod(self._rho.shape[1:])  # Total # of spatial bins (elements)
        meshes = np.meshgrid(*[np.arange(s) for s in self._rho.shape[1:]], indexing='ij')
        self.bin_indices = np.stack(meshes, axis=-1).reshape(-1, len(self._rho.shape[1:]))

        # Setup scaling factor:
        distance_scaling_factor = config.get('distance_scaling_factor', 1.0)
        inverse_scaling_factor = 1.0 / distance_scaling_factor
        self.dx *= inverse_scaling_factor                      # Proportional to m
        self._M /= inverse_scaling_factor                       # Proportional to m^-1
        self._L_phi /= inverse_scaling_factor
        self._k_laplacian *= np.pow(inverse_scaling_factor, 5)  # Proportional to m^5
        self._k_laplacian = self._k_laplacian.astype(self._float_type)

        self._inverse_scaling_factor = inverse_scaling_factor
        self._distance_scaling_factor = distance_scaling_factor
        self._use_autodiff = config.get('use_autodiff', False)  # Use manual derivative by default
        self._interface_scalar = config.get('interface_scalar', 1.)  # Scales interface energy, defaults to 1

        # Set the laplacian during init so don't need to check if every step
        self._cell_laplacian_fn = self._get_laplacian_function()
        self._dEdp = self._model.der_bulk_free_energy_autodiff if self._use_autodiff else self._model.der_bulk_free_energy

    def evolve(self, rho: Optional):
        raise NotImplementedError("evolve must be implemented by derived classes")

    '''@staticmethod
    def get_local_rho_species(rho, bin_indices):
        def get_rho_at_bin(indices):
            return rho[:, *indices]
        return jax.vmap(get_rho_at_bin)(bin_indices)'''

    @staticmethod
    def get_local_rho_species(rho, bin_indices):
        return rho[(slice(None),) + tuple(bin_indices.T)].T

    def _get_laplacian_function(self):
        # EVENTUALLY: Should look at other ways to do this periodic check potentially?
        return jax.jit(make_laplacian_roll(self._dim, self.dx))
