"""
Allen-Cahn implementation
"""
import jax.numpy as jnp
import jax
from functools import partial
from pfm.phase_field_models.phase_field_model import PhaseFieldModel


class AllenCahn(PhaseFieldModel):
    def __init__(self, free_energy_model, config, integrator, rng, custom_fn=None):
        # custom_fn is the initial condition function that might be specified
        # If it is None, the below super() is fine as either initial_density or load_from would
        # need to be specified (or an error will just occur)
        super().__init__(free_energy_model, config, integrator, rng, field_name="phi", custom_fn=custom_fn)
        self._L_phi = config.get('L_phi', 1.0)
        self.k = config.get('k', 1.0)

        # Set scaling if a distance_factor was specified (this is usually just multiplying by 1 however):
        self._L_phi /= self._inverse_scaling_factor   # Proportional to gamma^-1
        self.k *= self._inverse_scaling_factor ** 5   # Proportional to gamma^5

    def evolve(self, phi):
        return self.integrator.evolve(phi)

    @partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the average free energy per bin for the Allen-Cahn model.
        Assumes: phi.shape = (N_species, Nx, Ny) or similar
        """
        num_bins = jnp.prod(jnp.array(phi.shape[1:], dtype=jnp.int32))

        # 1. Calculate gradient contribution
        all_gradients = self.gradient(phi)
        sum_sq_grad = jnp.sum(all_gradients ** 2, axis=1)
        interfacial_density = 0.5 * self.k * jnp.sum(sum_sq_grad, axis=0)

        # 2. Calculate bulk contribution
        bulk_density = self.free_energy_model.bulk_free_energy(phi)

        # 3. Combine and average
        total_free_energy_density = bulk_density + interfacial_density
        total_fe = jnp.sum(total_free_energy_density) * self.V_bin
        avg_fe = total_fe / num_bins

        return avg_fe


