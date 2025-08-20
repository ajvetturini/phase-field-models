"""
Cahn-Hilliard implementation
"""
import jax.numpy as jnp
import jax
from functools import partial
from pfm.phase_field_models.phase_field_model import PhaseFieldModel


class CahnHilliard(PhaseFieldModel):
    def __init__(self, free_energy_model, config, integrator, rng, custom_fn=None):
        # custom_fn is the initial condition function that might be specified
        # If it is None, the below super() is fine as either initial_density or load_from would
        # need to be specified (or an error will just occur)
        super().__init__(free_energy_model, config, integrator, rng, field_name='rho', custom_fn=custom_fn)
        self.k_laplacian = config.get('k', 1.0)
        self.M = config.get('M', 1.0)
        self.dx = config.get('dx', 1.0)

        # Set scaling if a distance_factor was specified (this is usually just multiplying by 1 however):
        self.M /= self._inverse_scaling_factor  # Proportional to M^-1
        self.k_laplacian *= self._inverse_scaling_factor ** 5  # Proportional to M^5

    @partial(jax.jit, static_argnums=(0,))
    def evolve(self, rho):
        return self.integrator.evolve(rho)

    @partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, rho: jnp.ndarray) -> jnp.ndarray:
        """ Average free energy per bin """
        # Gradients: shape (N_species, N_dim, Nx, Ny, ...)
        all_gradients = self.gradient(rho)

        # Square + sum over dimensions, then sum over species
        # -> shape (Nx, Ny, ...)
        interfacial_density = self.k_laplacian * jnp.sum(jnp.sum(all_gradients ** 2, axis=1), axis=0)

        # Extract bin-wise species densities: shape (N_bins, N_species)
        bin_indices = self.integrator.bin_indices
        local_rhos_per_bin = self.integrator.get_local_rho_species(rho, bin_indices)

        # Bulk free energy per bin (scalar)
        bulk_per_bin = jax.vmap(self.free_energy_model.bulk_free_energy)(local_rhos_per_bin)  # (N_bins,)

        # Instead of reshaping, accumulate directly
        total_bulk = jnp.sum(bulk_per_bin)

        # Interfacial contribution: sum over grid directly
        total_interfacial = jnp.sum(interfacial_density)

        # Combine, multiply by volume element
        total_fe = (total_bulk + total_interfacial) * self.V_bin

        # Normalize by number of bins
        N_bins = local_rhos_per_bin.shape[0]
        avg_fe = total_fe / N_bins
        return avg_fe