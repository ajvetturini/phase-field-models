from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
from pfm.utils.delta import Delta

class SimpleWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)
        wertheim_config = config.get('wertheim')
        self._B2 = wertheim_config.get('B2')
        self._valence = int(wertheim_config.get('valence'))
        self._delta = Delta(wertheim_config.get('delta'))
        self._regularisation_delta = wertheim_config.get('regularisation_delta', self._floor_safety)

        # Scale constants if necesary:
        self._B2 *= (self._inverse_scaling_factor**3)
        self._delta.delta *= (self._inverse_scaling_factor**3)
        self._regularisation_delta *= (self._inverse_scaling_factor**3)

        self._log_delta = jnp.log(self._regularisation_delta)
        self._two_valence_delta = 2.0 * self._valence * self._delta.delta


    def N_species(self):
        # For a simple Wertheim implementation we will consider just 1 species
        return 1

    def _X(self, rho):
        """ Calculates fraction of molecules that are bonded (or unbounded) using the valence delta specified """
        rho_safe = jnp.maximum(rho, self._floor_safety)
        denom = self._two_valence_delta * rho_safe
        return (-1.0 + jnp.sqrt(1.0 + 2.0 * self._two_valence_delta * rho)) / denom

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
    def der_bulk_free_energy(self, rho):
        # rho_all_species has shape (1, Nx, Ny)
        der_f_ref = jnp.where(
            rho < self._regularisation_delta,
            rho / self._regularisation_delta + self._log_delta - 1.0,
            jnp.log(jnp.maximum(rho, 1e-12))  # Stability / safety so no nan
        )
        der_f_ref += (2 * self._B2 * rho)

        X = self._X(rho).astype(rho.dtype)  # Ensure _X is vectorized
        der_f_bond = jnp.where(
            rho > 0.,
            self._valence * jnp.log(X),  # Consider safety: jnp.log(jnp.maximum(X, 1e-9))?
            0.0,
        )

        return der_f_bond + der_f_ref

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

