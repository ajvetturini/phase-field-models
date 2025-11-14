import jax.numpy as jnp
import jax
from functools import partial
from pfm.phase_field_models.phase_field_model import PhaseFieldModel

class AllenCahn(PhaseFieldModel):
    def __init__(self, free_energy_model, config, integrator, rng, custom_fn=None):
        # custom_fn is the initial condition function that might be specified
        super().__init__(free_energy_model, config, integrator, rng, field_name="phi", custom_fn=custom_fn)
        self.k = config.get('k', 1.0)

        # Set scaling if a distance_factor was specified (this is usually just multiplying by 1 however):
        self.k *= self._inverse_scaling_factor ** 5   # Proportional to gamma^5

    def evolve(self, phi):
        return self.integrator.evolve(phi)

    @partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the average free energy per bin for the Allen-Cahn model.
        Assumes: phi.shape = (N_species, Nx, Ny) or similar
        """
        # Calculate gradient contribution
        all_gradients = self.gradient(phi)
        interfacial_density = 0.5 * self.k * jnp.sum(jnp.sum(all_gradients ** 2, axis=1), axis=0)  # k/2 * grad phi^2

        bin_indices = self.integrator.bin_indices
        local_phis_per_bin = self.integrator.get_local_rho_species(phi, bin_indices)

        # Calculate bulk contribution
        bulk_per_bin = jax.vmap(self.free_energy_model.bulk_free_energy)(local_phis_per_bin)
        total_bulk = jnp.sum(bulk_per_bin)
        total_interfacial = jnp.sum(interfacial_density)

        # Combine and average
        total_free_energy_density = total_bulk + total_interfacial
        total_fe = jnp.sum(total_free_energy_density) * self.V_bin

        return total_fe / local_phis_per_bin.shape[0]


