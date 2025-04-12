from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial

class Landau(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)  # First init the relevent FreeEnergyModel config options
        self._energy_config = config.get('landau')
        self._epsilon = jnp.array(self._energy_config.get('epsilon', 1.0), dtype=jnp.float64)

        if self._inverse_scaling_factor != 1.0:
            raise Exception('Landau free energy model does not support distance_scaling_factors from 1.0')


    def N_species(self):
        # For a simple Landau model, we assume a single component, N, that can have varying concentration.
        # If you're modeling a mixture with multiple conserved quantities, you'll need to adjust this
        return 1

    def bulk_free_energy(self, rho_species):
        op = rho_species[0]
        return -0.5 * self._epsilon * op**2 + 0.25 * op**4

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, rho_species):
        """ Calculates derivative of bulk free energy w.r.t. density of spatial grid. This is actually vmapped over
        the species, thus we can simply do:

        We must pass in rhos for other energy functions, it isn't used here though.
        """
        r = rho_species[0]  # Only 1 species in landau
        return -self._epsilon * r + r**3

    def _elementwise_bulk_free_energy(self, rho_species):
        """ Calculates the bulk free energy for each point in the grid. """
        return -0.5 * self._epsilon * rho_species ** 2 + 0.25 * rho_species ** 4

    def _total_bulk_free_energy(self, rho_species):
        return jnp.sum(self._elementwise_bulk_free_energy(rho_species))

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, species, rho_species):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        elementwise_grad_fn = jax.grad(self._total_bulk_free_energy)(rho_species)
        return elementwise_grad_fn






