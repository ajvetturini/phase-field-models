import jax.numpy as jnp
import numpy as np
from pfm.integrators.base_integrator import Integrator
import jax
from functools import partial

# Make sure to use float64 for stability or else divergence D:
jax.config.update("jax_enable_x64", True)

class jExplicitEuler(Integrator):

    def __init__(self, model, config):
        super().__init__(model, config)
        self._N_per_dim_minus_one = self._N_per_dim - 1
        self._log2_N_per_dim = int(jnp.log2(self._N_per_dim))
        self._dx = jnp.array(self._dx, dtype=jnp.float64)
        self._dt = jnp.array(self._dt, dtype=jnp.float64)

    def _flattened_field_to_grid(self, field_flat):
        """field_flat: shape (N_bins, N_species) → (N_species, Nx, Ny)"""
        shape = [self._model.N_species()] + [self._N_per_dim] * self._dim
        return field_flat.T.reshape(shape)

    @partial(jax.jit, static_argnums=(0,))
    def _evolve_jax(self, rho):
        """
        Explicit Euler step implemented using jax functionalities.

        Computes:
        rho(t+dt) = rho(t) + M * Δ (∂F/∂ρ) * dt
        """
        # Create rho_der array to store time derivatives
        rho_der = jnp.zeros_like(rho, dtype=jnp.float64)


        def update_rho_der(carry, idx):
            rho_der = carry
            coords = self._fill_coords(idx)
            rho_species = rho[tuple([slice(None)] + coords)]  # get species at index

            # Nested loop over species (replacing second loop)
            def update_species(carry, species):
                rho_der = carry
                drho = (self._model.der_bulk_free_energy(species, rho_species) - 2 * self._k_laplacian
                        * self._cell_laplacian(rho, species, coords))
                rho_der = rho_der.at[tuple([species] + coords)].set(drho)
                return rho_der, None

            rho_der, _ = jax.lax.scan(update_species, rho_der, jnp.arange(self._model.N_species()))

            return rho_der, None

        rho_der, _ = jax.lax.scan(update_rho_der, rho_der, jnp.arange(self._N_bins))

        def update_species(carry, species_idx):
            rho, coords = carry
            update = self._M * self._cell_laplacian(rho_der, species_idx, coords) * self._dt
            rho = rho.at[tuple([species_idx] + coords)].add(update)
            return (rho, coords), None

        # Define a function to update rho for each bin
        def update_bin(carry, bin_idx):
            rho = carry
            coords = self._fill_coords(bin_idx)  # get the coordinates for this bin
            # Use scan to loop over species for each bin
            (rho, _), _ = jax.lax.scan(update_species, (rho, coords), jnp.arange(self._model.N_species()))
            return rho, None

        # Use scan to loop over bins
        rho, _ = jax.lax.scan(update_bin, rho, jnp.arange(self._N_bins))

        return rho



    def _evolve_numpy_debug(self, rho):
        # Create rho_der array to store time derivatives
        rho_der = np.zeros_like(rho)

        # Calculate time derivatives first
        for idx in range(self._N_bins):
            coords = self._fill_coords(idx)
            rho_species = self._rho[tuple([slice(None)] + coords)]  # get species at index
            for species in range(self._model.N_species()):
                rho_der[tuple([species] + coords)] = (
                        self._model.der_bulk_free_energy(species, rho_species) - 2 * self._k_laplacian *
                        self._cell_laplacian(self._rho, species, coords)
                )

        # Integrate time derivatives
        for idx in range(self._N_bins):
            coords = self._fill_coords(idx)
            for species in range(self._model.N_species()):
                rho[tuple([species] + coords)] += (
                        self._M * self._cell_laplacian(rho_der, species, coords) * self._dt
                )

        return rho

    def evolve(self, rho):
        rho_jax = self._evolve_jax(rho)

        if jnp.isnan(rho_jax).any():
            raise Exception('ERROR: NaN found in rho')
        return rho_jax


    @partial(jax.jit, static_argnums=(0,))
    def _fill_coords(self, idx,):
        coords = []
        for d in range(self._dim):
            coords.append(idx & self._N_per_dim_minus_one)
            idx >>= self._log2_N_per_dim
        return coords

    @partial(jax.jit, static_argnums=(0,))
    def _cell_laplacian(self, field, species, coords,):
        if self._dim == 1:
            idx_m = (coords[0] - 1 + self._N_bins) & self._N_per_dim_minus_one
            idx_p = (coords[0] + 1) & self._N_per_dim_minus_one
            return (field[species, idx_m] + field[species, idx_p] - 2.0 * field[species, coords[0]]) / (
                    self._dx * self._dx)
        elif self._dim == 2:
            coords_xmy = [(coords[0] - 1 + self._N_bins) & self._N_per_dim_minus_one, coords[1]]
            coords_xym = [coords[0], (coords[1] - 1 + self._N_bins) & self._N_per_dim_minus_one]
            coords_xpy = [(coords[0] + 1) & self._N_per_dim_minus_one, coords[1]]
            coords_xyp = [coords[0], (coords[1] + 1) & self._N_per_dim_minus_one]
            return (field[species, coords_xmy[0], coords_xmy[1]] + field[species, coords_xpy[0], coords_xpy[1]] +
                    field[species, coords_xym[0], coords_xym[1]] + field[species, coords_xyp[0], coords_xyp[1]] -
                    4 * field[species, coords[0], coords[1]]) / (self._dx * self._dx)

        # Add 3D implementation if needed
        else:
            raise NotImplementedError("cell_laplacian not implemented for this dimension")
