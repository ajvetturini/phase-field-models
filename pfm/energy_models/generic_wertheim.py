from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
from pfm.utils.constants import Species, PatchInteraction
from pfm.utils.delta import Delta
from typing import List
import numpy as np

class GenericWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)
        wertheim_config = config.get("generic_wertheim")
        float_type = jnp.float64 if config.get("float_type", "float64") else jnp.float32
        if config.get('use_autodiff', False):
            raise ValueError("ERROR: Autodiff is current not supported for GenericWertheim as it relies on a while "
                             "loop to converge X (fraction of unbounded states) during updates")

        if wertheim_config is None:
            raise ValueError("Missing [generic_wertheim] in config")

        # Species setup
        self.species: List[Species] = []
        unique_patches_set = set()
        for idx, spec_cfg in enumerate(wertheim_config.get("species", [])):
            patches = spec_cfg.get("patches", [])
            new_species = Species(idx, patches)
            self.species.append(new_species)
            unique_patches_set.update(patches)

        if not self.species:
            raise ValueError("Missing 'generic_wertheim.species' array")
        self._Ns = len(self.species)

        self._unique_patch_ids = sorted(list(unique_patches_set))
        self._N_patches = max(self._unique_patch_ids) + 1

        # Delta interactions
        _delta_list = [0.0] * (self._N_patches * self._N_patches)
        deltas = wertheim_config.get("deltas", [])
        if not deltas:
            raise ValueError("Missing 'generic_wertheim.deltas' array")

        for elem in deltas:
            interaction = elem.get("interaction")
            patch_A, patch_B = self._parse_interaction(interaction, context="delta")

            # Equivalent of
            my_delta = Delta(elem)
            my_delta_val = my_delta.delta * (self._inverse_scaling_factor ** 3)

            _delta_list[patch_A * self._N_patches + patch_B] = my_delta_val
            _delta_list[patch_B * self._N_patches + patch_A] = my_delta_val

        # B2 coefficients
        species_count = self.N_species()
        _B2_list = [0.0] * (species_count * species_count)
        B2s = wertheim_config.get("B2s", [])
        if not B2s:
            raise ValueError("Missing 'generic_wertheim.B2s' array")

        expected_B2 = species_count * (species_count + 1) // 2
        if len(B2s) != expected_B2 and not wertheim_config.get("allow_unspecified_B2s", False):
            raise ValueError(f"Number of B2s ({len(B2s)}) does not match expected ({expected_B2})")

        for elem in B2s:
            interaction = elem.get("interaction")
            species_A, species_B = self._parse_interaction(interaction, context="B2")

            my_B2 = elem.get("value") * (self._inverse_scaling_factor ** 3)
            _B2_list[species_A * species_count + species_B] = my_B2
            _B2_list[species_B * species_count + species_A] = my_B2

        # Build patch interactions
        self._unique_patch_interactions: List[List[PatchInteraction]] = [[] for _ in range(self._N_patches)]
        for patch in self._unique_patch_ids:
            patch_interacting_species = []
            for other in self.species:
                interaction = PatchInteraction()
                for other_patch in other.unique_patches:
                    if _delta_list[patch * self._N_patches + other_patch.idx] > 0.0:
                        interaction.species = other.idx
                        interaction.patches.append(other_patch)
                if interaction.species != -1:
                    patch_interacting_species.append(interaction)
            self._unique_patch_interactions[patch] = patch_interacting_species

        # Convert  to jax arrays
        self._B2 = jnp.array(_B2_list).reshape((species_count, species_count))
        self.delta_matrix = jnp.array(_delta_list).reshape((self._N_patches, self._N_patches))
        patch_mult_matrix = np.zeros((species_count, self._N_patches))
        for spec in self.species:
            for patch in spec.unique_patches:
                patch_mult_matrix[spec.idx, patch.idx] = patch.multiplicity
        self.patch_multiplicity_matrix = jnp.array(patch_mult_matrix)
        target_patches, source_species, source_patches, multiplicities, deltas = [], [], [], [], []

        for patch_i, interactions in enumerate(self._unique_patch_interactions):
            for interaction in interactions:
                species_k = interaction.species
                for other_patch in interaction.patches:
                    patch_j = other_patch.idx

                    # For each elemental interaction, append its properties to lists
                    target_patches.append(patch_i)
                    source_species.append(species_k)
                    source_patches.append(patch_j)
                    multiplicities.append(other_patch.multiplicity)
                    deltas.append(self.delta_matrix[patch_i, patch_j])
        self.target_patches = jnp.array(target_patches, dtype=jnp.int32)
        self.source_species = jnp.array(source_species, dtype=jnp.int32)
        self.source_patches = jnp.array(source_patches, dtype=jnp.int32)
        self.multiplicities = jnp.array(multiplicities)
        self.deltas = jnp.array(deltas)

    @staticmethod
    def _parse_interaction(int_string, context):
        """Parse an interaction string like '3-7' into a tuple of ints."""
        parts = int_string.split("-")
        if len(parts) != 2:
            raise ValueError(f"The following {context} interaction specifier is malformed: {int_string}")

        try:
            part_A = int(parts[0])
            part_B = int(parts[1])
        except ValueError:
            raise ValueError(f"The following {context} interaction specifier has non-integer parts: {int_string}")

        return part_A, part_B

    def N_species(self):
        return self._Ns

    @partial(jax.jit, static_argnums=(0,))
    def _update_X(self, rhos, tolerance=1e-8, max_iter=10000):
        """Iteratively solves for the fraction of non-bonded sites, X."""
        # Initial state for the while_loop: (iteration, max_delta, X_vector)
        # Start with X=1, a common initial guess for this kind of problem.
        initial_X = jnp.ones(self._N_patches)
        init_state = (0, jnp.inf, initial_X)

        def cond_fun(state):
            """Continue loop if tolerance and max_iter are not met."""
            iteration, max_delta, _ = state
            return (max_delta > tolerance) & (iteration < max_iter)

        def body_fun(state):
            """Perform one iteration of the fixed-point solver."""
            iteration, _, current_X = state

            # Gather the relevant rhos and X values for each interaction
            rhos_per_interaction = rhos[self.source_species]
            X_per_interaction = current_X[self.source_patches]

            # Calculate all interaction terms in a single vectorized step
            terms = self.multiplicities * rhos_per_interaction * X_per_interaction * self.deltas
            patch_sums = jax.ops.segment_sum(terms, self.target_patches, num_segments=self._N_patches)

            # Calculate new X vector and the change to compare to stop condition
            new_X = 1.0 / (1.0 + patch_sums)
            max_delta = jnp.max(jnp.abs(new_X - current_X))

            return (iteration + 1, max_delta, new_X)

        # Run the while_loop to find the converged X
        _, _, final_X = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return final_X

    @partial(jax.jit, static_argnums=(0,))
    def bonding_energy(self, rhos):
        """ Calculates the bonding contribution to the free energy """
        # Calculate the per-patch free energy contribution
        Xs = self._update_X(rhos)
        patch_fe_contrib = jnp.log(Xs) - Xs / 2.0 + 0.5

        # Calculate the total free energy for each species
        species_fe = self.patch_multiplicity_matrix @ patch_fe_contrib

        # Calculate the final bonding free energy
        bonding_fe = jnp.dot(rhos, species_fe)
        return bonding_fe

    @partial(jax.jit, static_argnums=(0,))
    def bulk_free_energy(self, rhos):
        rtot = jnp.sum(rhos)
        x = rhos / rtot
        mixing_s = jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0.0))
        B2_contrib = rhos @ self._B2 @ rhos  # quadratic form since B2 is symmetric matrix

        log_rho_term = jnp.where(rtot > 0, rtot * (jnp.log(rtot * self._density_conversion_factor) - 1.0), 0.0)
        f_ref = log_rho_term + rtot * mixing_s + B2_contrib

        return f_ref + self.bonding_energy(rhos)

    def _der_bulk_free_energy_single(self, rhos):
        """ Calculates the derivative of the bulk free energy for a single point in the grid. """
        # Calculate the derivative of the reference free energy (f_ref)
        Xs = self._update_X(rhos)
        b2_contrib = 2.0 * (self._B2 @ rhos)
        der_f_ref = jnp.where(rhos > 0, jnp.log(rhos), 0.0) + b2_contrib
        der_f_bond = self.patch_multiplicity_matrix @ jnp.log(Xs)

        # Combine and apply zero-density guard
        total_der = der_f_ref + der_f_bond
        return jnp.where(rhos > 0, total_der, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, rhos):
        rhos_flat = jnp.moveaxis(rhos, 0, -1).reshape(-1, rhos.shape[0])
        der_bulk_per_bin = jax.vmap(self._der_bulk_free_energy_single)(rhos_flat)
        return der_bulk_per_bin.T.reshape(rhos.shape)


    def _der_bulk_free_energy_point_autodiff(self, rhos):
        """ Calculates the bulk free energy for each point in the grid. """
        return jax.grad(self.bulk_free_energy)(rhos)

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, rhos):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        # rhos is shape (N_species, Nx, Ny)
        rhos_flat = jnp.moveaxis(rhos, 0, -1).reshape(-1, rhos.shape[0])
        out = jax.vmap(self._der_bulk_free_energy_point_autodiff)(rhos_flat)
        return out.T.reshape(rhos.shape)