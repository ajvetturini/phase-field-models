from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
from pfm.utils.delta import Delta

class SimpleWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)

        # Simple Wetherim specific properties. Note that these come from a 2nd expansion of Virial Coefficient:
        wertheim_config = config.get('wertheim')
        self._B2 = wertheim_config.get('B2')
        self._valence = int(wertheim_config.get('valence'))
        self._delta = Delta(wertheim_config.get('delta'))
        self._regularisation_delta = wertheim_config.get('regularisation_delta', 1e-9)  # Default to a small value ~0

        # Scale constants if necesary:
        self._B2 *= (self._inverse_scaling_factor**3)
        self._delta.delta *= (self._inverse_scaling_factor**3)   # Need to access actual value of delta (_delta.delta)
        self._regularisation_delta *= (self._inverse_scaling_factor**3)

        self._log_delta = jnp.log(self._regularisation_delta)
        self._two_valence_delta = 2.0 * self._valence * self._delta.delta


    def N_species(self):
        # For a simple Landau model, we assume a single component, N, that can have varying concentration.
        # If you're modeling a mixture with multiple conserved quantities, you'll need to adjust this
        return 1

    def _X(self, rho):
        """ Calculates fraction of molecules that are bonded (or unbounded) using the valence delta specified """
        return (-1.0 + jnp.sqrt(1.0 + 2.0 * self._two_valence_delta * rho)) / (self._two_valence_delta * rho)

    @partial(jax.jit, static_argnums=(0,))
    def bulk_free_energy(self, rho_species):
        r0 = rho_species[0]  # Only 1 species in SimpleWertheim
        rho_sqr = r0 * r0  # Calculate square once

        # Reference energy:
        f_ref = jnp.where(r0 < self._regularisation_delta,  # Wherever order param < 0 (gas) use the regularisation
                          rho_sqr / (2.0 * self._regularisation_delta) + (
                                  r0 * self._log_delta - self._regularisation_delta / 2.0
                          ),
                          r0 * jnp.log(r0 * self._density_conversion_factor))  # Otherwise in liquid us r0

        f_ref += -r0 + self._B2 * rho_sqr

        # Bond energy:
        f_bond = jnp.where(r0 > 0.0,
                           self._valence * r0 * (jnp.log(self._X(r0)) + 0.5 * (1.0 - self._X(r0))),
                           0.0
                           )

        return f_ref + f_bond

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, rho_species):
        """ Calculates derivative of bulk free energy w.r.t. density of spatial grid. This is actually vmapped over
        the species.
        """
        rho = rho_species[species]
        der_f_ref = jnp.where(
            rho < self._regularisation_delta,
            rho / self._regularisation_delta + self._log_delta - 1.0,
            jnp.log(rho)
        )
        der_f_ref += 2 * self._B2 * rho

        X = self._X(rho)
        der_f_bond = jnp.where(
            rho > 0.,
            self._valence * jnp.log(X),
            0.0,
        )
        return der_f_ref + der_f_bond

    def _elementwise_bulk_free_energy(self, r0):
        """ Calculates the bulk free energy for each point in the grid. """
        rho_sqr = r0 * r0  # Calculate square once
        Xr = self._X(r0)
        # Reference energy:
        f_ref = jnp.where(r0 < self._regularisation_delta,  # Wherever order param < 0 (gas) use the regularisation
                          rho_sqr / (2.0 * self._regularisation_delta) + (
                                  r0 * self._log_delta - self._regularisation_delta / 2.0
                          ),
                          r0 * jnp.log(r0 * self._density_conversion_factor))  # Otherwise in liquid us r0

        f_ref += -r0 + self._B2 * rho_sqr

        # Bond energy:
        f_bond = jnp.where(r0 > 0.0,
                           self._valence * r0 * (jnp.log(Xr) + 0.5 * (1.0 - Xr)),
                           0.0
                           )

        return f_ref + f_bond

    def _total_bulk_free_energy(self, rho_species):
        return jnp.sum(self._elementwise_bulk_free_energy(rho_species))

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, species, rho_species):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        elementwise_grad_fn = jax.grad(self._total_bulk_free_energy)(rho_species)
        return elementwise_grad_fn


