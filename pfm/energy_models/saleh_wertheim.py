from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
from pfm.utils.delta import Delta

class SalehWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)

        # Saleh Wertheim specific properties
        saleh_config = config.get('saleh')
        self._B2 = saleh_config.get('B2')
        self._B3 = saleh_config.get('B3', 0.)
        self._valence = saleh_config.get('valence')
        self._linker_half_valence = self._valence[2] / 2

        # Read in the Deltas from the config:
        self._delta_AA = Delta(saleh_config.get('delta_AA'))
        self._delta_BB = Delta(saleh_config.get('delta_BB'))

        # Scale values
        self._B2 *= (self._inverse_scaling_factor ** 3)
        self._B3 *= (self._inverse_scaling_factor ** 3)
        self._delta_AA.delta *= (self._inverse_scaling_factor ** 3)
        self._delta_BB.delta *= (self._inverse_scaling_factor ** 3)

    def N_species(self):
        # Saleh uses 3 species, please see GenericWertheim for a more generalized implementation
        return 3

    def bonding_energy(self, total_rhos):
        # Note that this term is specific to the Saleh system and
        rho_factor = self._delta_AA.delta * (self._valence[0] * total_rhos[0] +
                                             self._linker_half_valence * total_rhos[2])
        X1A = (-1.0 + jnp.sqrt(1.0 + 4.0 * rho_factor)) / (2.0 * rho_factor)
        part1 = jnp.log(X1A) - X1A / 2.0 + 0.5

        rho_factor2 = self._delta_BB.delta * (self._valence[1] * total_rhos[1] +
                                              self._linker_half_valence * total_rhos[2])
        X2b = (-1.0 + jnp.sqrt(1.0 + 4.0 * rho_factor2)) / (2.0 * rho_factor2)
        part2 = jnp.log(X2b) - X2b / 2.0 + 0.5

        bonding_fe = ((total_rhos[0] * self._valence[0] * part1) +
                      (total_rhos[1] * self._valence[1] * part2) +
                      (total_rhos[2] * self._linker_half_valence * (part1 + part2))
        )
        return bonding_fe

    @partial(jax.jit, static_argnums=(0,))
    def bulk_free_energy(self, rhos):
        rtot = jnp.sum(rhos)

        # mixing entropy
        x = rhos / rtot
        mixing_s = jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0.0))

        # Compile virial coefficients and then total bulk energy as reference + bonding:
        B2 = self._B2 * rtot
        B3 = 0.5 * self._B3 * rtot**2

        f_ref = rtot * (jnp.log(rtot * self._density_conversion_factor) - 1.0 + mixing_s + B2 + B3)

        return f_ref + self.bonding_energy(rhos)

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, rhos):
        rho_tot = jnp.sum(rhos)

        # reference part
        epsilon = 1e-20
        safe_rhos = jnp.maximum(rhos, epsilon)
        der_f_ref = jnp.log(safe_rhos) + 2.0 * self._B2 * rho_tot + 3.0 * self._B3 * rho_tot ** 2

        # bonding part
        der_f_bond = jnp.zeros_like(rhos)
        contribution_species_0 = self._der_contribution(safe_rhos, 0)
        contribution_species_1 = self._der_contribution(safe_rhos, 1)
        der_f_bond = der_f_bond.at[0].set(self._valence[0] * contribution_species_0)
        der_f_bond = der_f_bond.at[1].set(self._valence[1] * contribution_species_1)
        der_f_bond = der_f_bond.at[2].set(self._linker_half_valence * (contribution_species_0 + contribution_species_1))

        # Combine and if a species has ~0 density, its chemical potential derivative will be 0:
        total_der_bulk = der_f_ref + der_f_bond
        safe_der_bulk = jnp.where(rhos > 0, total_der_bulk, 0.0)
        return safe_der_bulk

    def _der_contribution(self, rhos, species):
        if species == 0:
            delta = self._delta_AA.delta
        else:
            delta = self._delta_BB.delta

        rho_factor = delta * (self._valence[species] * rhos[species] +
                              self._linker_half_valence * rhos[2])

        X = (-1.0 + jnp.sqrt(1.0 + 4.0 * rho_factor)) / (2.0 * rho_factor)
        return jnp.where(rho_factor >= 0, jnp.log(X), 0.0)

    def _der_bulk_free_energy_point_autodiff(self, rhos):
        """ Calculates the bulk free energy for each point in the grid. """
        return jax.grad(self.bulk_free_energy)(rhos)

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, rhos):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        rhos_flat = jnp.moveaxis(rhos, 0, -1).reshape(-1, rhos.shape[0])
        out = jax.vmap(self._der_bulk_free_energy_point_autodiff)(rhos_flat)
        return out.T.reshape(rhos.shape)