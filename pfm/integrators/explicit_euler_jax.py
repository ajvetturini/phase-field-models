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

        # Check stability of hyperparameters:
        k = 2 * self._M * self._k_laplacian
        lhs = k * self._dt / self._dx**4

        if config.get('verbose', True):
            print('Explicit Euler Integrator specified')
            print(f'Stability term: {lhs} | (This term should be << 1 or else Explicit Euler will be unstable.)')

    @partial(jax.jit, static_argnums=(0,))
    def _evolve_jax(self, rho):
        """
        Explicit Euler step implemented using jax functionalities.

        Computes:
        rho(t+dt) = rho(t) + M * Δ (∂F/∂ρ) * dt
        """
        # Compute dF/dρ per species and bin
        dF_dRho = jax.vmap(self._model.der_bulk_free_energy, in_axes=(0, 0))(
            jnp.arange(self._model.N_species()), rho
        )  # shape: (N_species, Nx, Ny)

        lap_rho = self._cell_laplacian(rho)  # shape: (N_species, Nx, Ny)
        #d_rho = dF_dRho - 2.0 * self._k_laplacian * lap_rho
        d_rho = dF_dRho - self._k_laplacian * lap_rho

        lap_d_rho = self._cell_laplacian(d_rho)  # shape: (N_species, Nx, Ny)

        return rho + self._M * lap_d_rho * self._dt


    def evolve(self, rho):
        rho_jax = self._evolve_jax(rho)

        if jnp.isnan(rho_jax).any():
            raise Exception('ERROR: NaN found in rho')
        return rho_jax

    @partial(jax.jit, static_argnums=(0,))
    def _cell_laplacian(self, phi):
        """
        Periodic boundary conditions LaPlacian (uses roll below)

        phi: shape (N_species, Nx, ...)
        Returns: shape (N_species, Nx, ...)
        """
        Nx, Ny, Nz = self._N_per_dim, self._N_per_dim, self._N_per_dim,
        if self._dim == 1:
            left = jnp.roll(phi, shift=+1, axis=1)
            right = jnp.roll(phi, shift=-1, axis=1)
            center = phi
            return (left + right - 2 * center) / self._dx ** 2

        elif self._dim == 2:
            up = jnp.roll(phi, shift=+1, axis=1)
            down = jnp.roll(phi, shift=-1, axis=1)
            left = jnp.roll(phi, shift=+1, axis=2)
            right = jnp.roll(phi, shift=-1, axis=2)
            center = phi

            # Assume box-like system (dx = dy = dz)
            return (up + down + left + right - 4 * center) / (self._dx ** 2)


        else:
            raise NotImplementedError("Only 1D and 2D Laplacians implemented.")



def _test_cell_laplacian():
    class dummy_model:
        def N_species(self):
            return 1

    dm = dummy_model()
    dummy_integrator = jExplicitEuler(dm, {})

    # Verify that a constant row field has 0 laplacian everywhere
    const = 3.14
    rho = jnp.full((1, 32, 32), const)  # One species, 32x32 grid
    lap = dummy_integrator._cell_laplacian(rho)
    assert jnp.allclose(lap, 0.0), "Laplacian of constant field should be zero."

    # Test 1D sinusoidal:
    N = 64  #  NOTE: IF N is smaller (e.g., 64) then this might FAIL!
    L = 1.0
    dx = L / N
    k = 2 * jnp.pi / L
    x = jnp.arange(N) * dx
    rho = jnp.sin(k * x)[None, :]  # shape (1, N)

    dummy_integrator._dx = dx
    dummy_integrator._dim = 1
    lap = dummy_integrator._cell_laplacian(rho)

    expected = -k ** 2 * jnp.sin(k * x)[None, :]
    error = jnp.max(jnp.abs(lap - expected))
    assert error < 1e-2, f"Laplacian error too large: {error}"

    # Test 2D Sinusoidal
    N = 64                # Smaller n (e.g., 64 or 128) fails the check due to 1st order approx
    L = 1.0
    dx = L / N
    k = 2 * jnp.pi / L
    x = jnp.arange(N) * dx
    y = jnp.arange(N) * dx
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    rho_2d = jnp.sin(k * X) * jnp.sin(k * Y)  # shape (N, N)
    rho = rho_2d[None, :, :]  # shape (1, Nx, Ny)

    dummy_integrator._dim = 2
    dummy_integrator._dx = dx

    lap = dummy_integrator._cell_laplacian(rho)

    expected = -2 * k ** 2 * rho
    error = jnp.max(jnp.abs(lap - expected))
    assert error < 1e-2, f"Laplacian error too large: {error}"


if __name__ == '__main__':
    _test_cell_laplacian()
