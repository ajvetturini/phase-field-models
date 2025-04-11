"""
Cahn-Hilliard implementation
"""
import jax.numpy as jnp
import jax
from functools import partial
from pfm.models.phase_field_model import PhaseFieldModel
jax.config.update("jax_enable_x64", True)

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

    def evolve(self, rho):
        return self.integrator.evolve(rho)

    '''@partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the average free energy per bin, including bulk and interfacial contributions.
        Assumes: rho.shape = (N_species, Nx, Ny) or similar
        """
        num_bins = jnp.prod(jnp.array(rho.shape[1:], dtype=jnp.int32))  # Use int32 or int64

        # 1. Calculate interfacial contribution (vectorized)
        # Assuming vectorized_gradient returns shape (N_species, dim, Nx, Ny, ...)
        all_gradients = self.gradient(rho)

        # Sum of squares of gradient components (sum over 'dim' axis, assuming it's axis 1)
        sum_sq_grad = jnp.sum(all_gradients ** 2, axis=1)  # Shape: (N_species, Nx, Ny, ...)
        total_interfacial_density = self.k_laplacian * jnp.sum(sum_sq_grad, axis=0)  # Shape: (Nx, Ny, ...)

        # 2. Calculate bulk contribution using vectorized operations
        bulk_density = self.free_energy_model.bulk_free_energy(rho)

        # 3. Combine and average
        total_free_energy_density = bulk_density + total_interfacial_density  # Shape: (Nx, Ny, ...)
        total_fe = jnp.sum(total_free_energy_density) * self.V_bin
        avg_fe = total_fe / num_bins

        return avg_fe
'''

    @partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, rho: jnp.ndarray) -> jnp.ndarray:
        """ Calculates the per-bin average free energy of the order parameter rho """
        # 1. Calculate interfacial contribution (vectorized)
        # Assuming vectorized_gradient returns shape (N_species, dim, Nx, Ny, ...)
        all_gradients = self.gradient(rho)

        # Sum of squares of gradient components (sum over 'dim' axis, assuming it's axis 1)
        sum_sq_grad = jnp.sum(all_gradients ** 2, axis=1)  # Shape: (N_species, Nx, Ny, ...)
        total_interfacial_density = self.k_laplacian * jnp.sum(sum_sq_grad, axis=0)

        bin_indices = self.integrator.bin_indices
        local_rhos_per_bin = self.integrator.get_local_rho_species(rho, bin_indices)  # Shape: (N_bins, N_species)
        Ns = self.free_energy_model.N_species()
        N_bins = local_rhos_per_bin.shape[0]

        # Compute dF/dρ per species and bin
        def bulk_term_per_bin(local_rho):
            return self.free_energy_model.bulk_free_energy(local_rho)

        bulk_term_scalar = jax.vmap(bulk_term_per_bin)(local_rhos_per_bin)  # vmap over bins
        bulk_density = bulk_term_scalar.reshape(rho.shape[1:])  # Re-shape back

        # 3. Combine contributions and multiply by volume of bin
        total_free_energy_density = bulk_density + total_interfacial_density  # Shape: (Nx, Ny, ...)
        total_fe = jnp.sum(total_free_energy_density) * self.V_bin

        # Get average energy over the number of bins:
        avg_fe = total_fe / N_bins
        return avg_fe
