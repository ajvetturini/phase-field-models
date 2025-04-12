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
    def bulk_free_energy(self, rho_species):
        axes_to_sum = tuple(range(1, rho_species.ndim))
        rho_species_total = jnp.sum(rho_species, axis=axes_to_sum)  # Get total density of each species in dim space
        rtot = jnp.sum(rho_species_total)  # Then get the total densities of all species together
        mixing_s = 0.  # Mixing entropy
        for i in range(self.N_species()):
            x_i = rho_species_total[i] / rtot  # mole fraction of i-th species
            mixing_s += jnp.where(x_i > 0, x_i * jnp.log(x_i), 0)

        # Compile virial coefficients and then total bulk energy as reference + bonding:
        B2 = self._B2 * rtot
        B3 = 0.5 * self._B3 * rtot**2

        f_ref = rtot * (jnp.log(rtot * self._density_conversion_factor) - 1.0 + mixing_s + B2 + B3)

        return f_ref + self.bonding_energy(rho_species_total)

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, rho_species):
        rho_species_val = rho_species[species]

        def calculate_derivative(carry):
            _species, _rho_species = carry
            _rho = jnp.sum(_rho_species)
            df_ref = jnp.log(_rho_species[_species]) + 2.0 * self._B2 * _rho + 3.0 * self._B3 * _rho ** 2

            def case_0(r):
                return self._valence[0] * self._der_contribution(r, 0)

            def case_1(r):
                return self._valence[1] * self._der_contribution(r, 1)

            def case_other(r):
                return self._linker_half_valence * (
                        self._der_contribution(r, 0) + self._der_contribution(r, 1)
                )

            df_bond = jax.lax.cond(
                _species == 0, case_0,
                lambda r2d2: jax.lax.cond(
                    _species == 1, case_1, case_other, _rho_species
                ), _rho_species
            )
            return df_ref + df_bond

        def return_zero(carry):
            return 0.0

        result = jax.lax.cond(rho_species_val == 0.0,
                              return_zero,
                              calculate_derivative,
                              (species, rho_species))
        return result

    def _der_contribution(self, rhos, species):
        if species == 0:
            delta = self._delta_AA.delta
        else:
            delta = self._delta_BB.delta

        rho_factor = delta * (self._valence[species] * rhos[species] + self._linker_half_valence * rhos[2])
        x = (-1. + jnp.sqrt(1. + 4. * rho_factor)) / (2. * rho_factor)

        ret_val = jax.lax.cond(
            rho_factor >= 0.,
            lambda: jnp.log(x),
            lambda: 0.
        )

        return ret_val

    def _elementwise_bulk_free_energy(self, species, rho_species):
        """ Calculates the bulk free energy for each point in the grid. """
        pass

    def _total_bulk_free_energy(self, rho_species):
        return jnp.sum(self._elementwise_bulk_free_energy(rho_species))

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, species, rho_species):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        elementwise_grad_fn = jax.grad(self._total_bulk_free_energy)(rho_species)
        return elementwise_grad_fn