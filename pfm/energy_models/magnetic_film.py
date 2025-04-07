""" Simple double well potential to verify the Allen Cahn implementation """
from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)

class MagneticFilm(FreeEnergyModel):  # Assuming you have a FreeEnergyModel base class

    def __init__(self, config):
        super().__init__(config)
        self._energy_config = config.get('magnetic_film')
        self._delta = jnp.array(self._energy_config.get('delta', 1.0), dtype=jnp.float64)

        self._autograd_fn = jax.jit(jax.grad(self._elementwise_bulk_free_energy))

    def N_species(self):
        return 1

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, rho_species):
        """ Derivative of the double-well bulk free energy. """
        return 12 * jnp.pow(rho_species, 2) - 4

    def _elementwise_bulk_free_energy(self, rho_species):
        """ Calculates the double-well bulk free energy for each point in the grid. """
        return 4 * self._delta * (jnp.pow(rho_species, 3) - rho_species)

    def _total_bulk_free_energy(self, rho_species):
        return jnp.sum(self._elementwise_bulk_free_energy(rho_species))

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, species, rho_species):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        elementwise_grad_fn = jax.grad(self._total_bulk_free_energy)(rho_species)
        return elementwise_grad_fn

    def bulk_free_energy(self, rho_species):
        op = rho_species[0]
        return 4 * self._delta * (jnp.pow(op, 3) - op)
