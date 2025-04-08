from pfm import SimulationManager
import toml
import time
import numpy as np
from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)

class MagneticFilm(FreeEnergyModel):
    def __init__(self, config):
        super().__init__(config)
        self._energy_config = config.get('magnetic_film')
        self._delta = jnp.array(self._energy_config.get('delta', 1.0), dtype=jnp.float64)

        self._autograd_fn = jax.jit(jax.grad(self._elementwise_bulk_free_energy))

    def N_species(self):
        return 1

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, phi):
        """ Derivative of the double-well bulk free energy. """
        return 4 * self._delta * (jnp.pow(phi, 3) - phi)

    def _elementwise_bulk_free_energy(self, phi_species):
        """ Calculates the double-well bulk free energy for each point in the grid. """
        return self._delta * (jnp.pow(phi_species, 4) - 2 * (jnp.pow(phi_species, 2)))

    def _total_bulk_free_energy(self, rho_species):
        return jnp.sum(self._elementwise_bulk_free_energy(rho_species))

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, species, rho_species):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        elementwise_grad_fn = jax.grad(self._total_bulk_free_energy)(rho_species)
        return elementwise_grad_fn

    def bulk_free_energy(self, rho_species):
        op = rho_species[0]  # Only 1 species
        return self._delta * (jnp.pow(op, 4) - 2 * jnp.pow(op, 2))


def custom_initial_condition(init_phi):
    # init_phi is a zeros array of the simulation field shape
    shape = init_phi.shape
    num_species = shape[0]
    grid_shape = shape[1:]

    # Note that there is a species in the 0th idx
    for i in range(num_species):
        # Generate random values for the current species on the spatial grid
        init_phi[i] = 2 * np.random.rand(*grid_shape) - 1  # Values between -1 and 1

    return init_phi

# NOTES
# Overall, this is a SHORT simulation! We are looking at the early separation (first few hundred timesteps)
# The system achieves a minimum relatively quickly (within few thousand steps), so if you run a system
# too long, the results file may not tell an accurate story.
c = toml.load('input_magnetic_film.toml')
manager = SimulationManager(c, custom_energy=MagneticFilm, custom_initial_condition=custom_initial_condition)
start = time.time()
manager.run_jax()
end = time.time() - start

minutes = int(end // 60)
seconds = int(end % 60)

print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')