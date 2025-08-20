import jax.numpy as jnp
from pfm.integrators.base_integrator import Integrator
import jax
from functools import partial

class ExplicitEuler(Integrator):

    def __init__(self, model, config):
        super().__init__(model, config)
        self._dx = jnp.array(self.dx, dtype=self._float_type)
        self._dt = jnp.array(self._dt, dtype=self._float_type)

        # Check (roughly) stability of hyperparameters for explicit euler
        k = 2 * self._M * self._k_laplacian
        lhs = k * self._dt / self._dx**4

        if config.get('verbose', True):
            print('Explicit Euler Integrator specified')
            if self._use_autodiff:
                print('Using autodiff for bulk energy derivative')
            print(f'Stability term: {lhs} | This term should be << 1 or else Explicit Euler will be unstable.')

        # In the init we set the evolve function, allowing us to easily change what data gets transferred to individual
        # energy phase_field_models:
        if config.get('model', 'ch').lower() == 'ch':
            self._evolve_fn = self._evolve_cahn_hilliard
        elif config.get('model', 'ac').lower() == 'ac':
            self._evolve_fn = self._evolve_allen_cahn
        else:
            raise Exception('Invalid integrator method, valid options are: `ch`, `ac`, ')

    @partial(jax.jit, static_argnums=(0,))
    def _evolve_cahn_hilliard(self, rho):
        def laplacian(phi):
            # NOTE: This is very slow specifically in float64 mode
            result = -2 * (phi.ndim - 1) * phi  # ndim - 1 to account for species dimension
            for axis in range(1, phi.ndim):  # skip species axis (0)
                result += (jnp.roll(phi, +1, axis) + jnp.roll(phi, -1, axis))
            return result / self._dx ** 2

        # Assuming _dEdp is vectorized over (N_species, Nx, Ny)
        bulk_energy = self._dEdp(rho)  # shape: (N_species, Nx, Ny)

        lap_rho = laplacian(rho)
        chemical_potential = bulk_energy - (self._interface_scalar * self._k_laplacian * lap_rho)
        lap_d_rho = laplacian(chemical_potential)

        return rho + self._M * lap_d_rho * self._dt

    @partial(jax.jit, static_argnums=(0, ))
    def _evolve_allen_cahn(self, phi):
        """
        Explicit Euler step for Allen-Cahn

        Computes:
        phi(t+dt) = phi(t) - l_phi * (dF/dphi) * dt
        where dF/dphi = d(bulk_free_energy)/d(phi) - k * laplacian(phi)
        Here, we assume the free energy model provides the derivative of the bulk term.
        We'll approximate k * laplacian(phi) using the _k_laplacian and our laplacian method.
        """
        # Get local rho_species for all bins
        local_rhos_per_bin = self.get_local_rho_species(phi, self.bin_indices)  # Shape: (N_bins, N_species)

        # Compute dF/dρ per species and bin
        def bulk_term_for_bin(local_rho):
            return jax.vmap(self._dEdp, in_axes=(0, None))(
                jnp.arange(self._model.N_species()), local_rho
            )

        bulk_term_scalar = jax.vmap(bulk_term_for_bin)(local_rhos_per_bin)  # vmap over bins
        bulk_energy = bulk_term_scalar.reshape((self._model.N_species(),) + phi.shape[1:])  # Re-shape back

        lap_phi = self._cell_laplacian(phi)  # shape: (N_species, Nx, Ny)
        interface_energy = self._interface_scalar * self._k_laplacian * lap_phi
        energy_difference = bulk_energy - interface_energy

        return phi - (self._L_phi * energy_difference * self._dt)

    def evolve(self, rho):
        """ The order parameter (rho, phi) is passed in and updated. We need to specify which derivative function
         to use if autodiff is enabled.
         """
        #if jnp.isnan(new_rho).any():  # Can't uncomment or jit will break
        #    raise Exception('ERROR: NaN found in rho update. This is likely due to numerical method diverging.')
        return self._evolve_fn(rho)


    def _cell_laplacian(self, phi):
        """
        Periodic boundary conditions LaPlacian (uses roll). This is defined during base Integrator class so we don't
        need to check the if conditionals constantly

        phi: shape (N_species, Ny, Nx,)
        Returns: shape (N_species, Ny, Nx)
        """
        return self._cell_laplacian_fn(phi)



def _test_cell_laplacian():
    class dummy_model:
        def N_species(self):
            return 1

    dm = dummy_model()
    dummy_integrator = ExplicitEuler(dm, {})

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
