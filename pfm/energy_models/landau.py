from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial

class Landau(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)
        self._energy_config = config.get('landau')
        float_type = jnp.float64 if config.get('float_type', 'float32') == 'float64' else jnp.float32
        self._epsilon = jnp.array(self._energy_config.get('epsilon', 1.0), dtype=float_type)

        if self._inverse_scaling_factor != 1.0:
            raise Exception('Landau free energy model does not support distance_scaling_factors from 1.0')


    def N_species(self):
        return 1

    def bulk_free_energy(self, rho_species):
        op = rho_species[0]
        return -0.5 * self._epsilon * op**2 + 0.25 * op**4

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_pointwise(self, rho_species):
        """Calculates derivative of bulk free energy w.r.t. density for a single point.
        Currently this is used in a PINN (not the numerical method solves) but hasn't been fully tested yet.
        """
        r = rho_species[0]
        df = -self._epsilon * r + r ** 3
        # Return shape (n_species,) = (1,)
        return jnp.array([df])

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, rho_species):
        """ Calculates derivative of bulk free energy w.r.t. density of spatial grid """
        r = rho_species[0]
        df = -self._epsilon * r + r**3
        # return in shape of (1, *grid_shape)
        return df[None, ...]

    def _der_bulk_free_energy_point_autodiff(self, rhos):
        return jax.grad(self.bulk_free_energy)(rhos)

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, rhos):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        rhos_flat = jnp.moveaxis(rhos, 0, -1).reshape(-1, rhos.shape[0])
        out = jax.vmap(self._der_bulk_free_energy_point_autodiff)(rhos_flat)
        return out.T.reshape(rhos.shape)






