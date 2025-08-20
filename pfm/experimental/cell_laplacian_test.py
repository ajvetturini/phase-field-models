# Using this to testbed what types of laplacian operators are useful

import jax
import jax.numpy as jnp
#jax.config.update("jax_enable_x64", True)
from functools import partial

@jax.jit
def _X(rho):
    """ Calculates fraction of molecules that are bonded (or unbounded) using the valence delta specified """
    rho_safe = jnp.maximum(rho, 1e-12)
    two_valence_delta = 173174.9910
    denom = two_valence_delta * rho_safe
    return (-1.0 + jnp.sqrt(1.0 + 2.0 * two_valence_delta * rho)) / denom

@jax.jit
def _der_bulk_free_energy(rho):
    # rho_all_species has shape (1, Nx, Ny)
    der_f_ref = jnp.where(
        rho < 1e-9,
        rho / 1e-9 + -21.7232,
        jnp.log(jnp.maximum(rho, 1e-9))  # Stability / safety so no nan
    )
    der_f_ref += 2 * 2190 * rho

    X = _X(rho)  # Ensure _X is vectorized
    der_f_bond = jnp.where(
        rho > 0.,
        4 * jnp.log(X),  # Consider safety: jnp.log(jnp.maximum(X, 1e-9))?
        0.0,
    )

    return der_f_bond + der_f_ref

@jax.jit
def _evolve_cahn_hilliard(rho):
    def laplacian(phi):
        result = -2 * (phi.ndim - 1) * phi
        for axis in range(1, phi.ndim):  # skip species axis (0)
            result += jnp.roll(phi, +1, axis) + jnp.roll(phi, -1, axis)
        return result / 10 ** 2

    # Assuming _dEdp is vectorized over (N_species, Nx, Ny)
    bulk_energy = _der_bulk_free_energy(rho)  # shape: (N_species, Nx, Ny)

    lap_rho = laplacian(rho)
    ##lap_rho = jnp.zeros(rho.shape)
    chemical_potential = bulk_energy - 1e6 * lap_rho
    lap_d_rho = laplacian(chemical_potential)
    ##lap_d_rho = jnp.zeros(chemical_potential.shape)

    return rho + lap_d_rho * 1e-5



@partial(jax.jit, static_argnums=(1,))
def _RUN(rho_0, steps):

    def _evolve(_r, _):
        _r = _evolve_cahn_hilliard(_r)  # Evolve state first
        return _r, None

    rho_n, _ = jax.lax.scan(_evolve, rho_0, jnp.arange(steps))
    return rho_n

if __name__ == '__main__':
    import time
    import numpy as np
    start = time.time()
    r0 = np.loadtxt('/home/aj/GitHub/phase-field-phase_field_models/Examples/Wertheim/simple_wertheim/jax/init_0.dat')
    _RUN(r0, 100000)

    end = time.time() - start
    minutes = int(end // 60)
    seconds = int(end % 60)
    print(f'RUN: {minutes} min {seconds} secs')