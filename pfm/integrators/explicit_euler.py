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
            # NOTE: This is very slow (specifically in float64 mode)
            result = -2 * (phi.ndim - 1) * phi  # ndim - 1 to account for species dimension
            for axis in range(1, phi.ndim):  # skip species axis (0)
                result += (jnp.roll(phi, +1, axis) + jnp.roll(phi, -1, axis))
            return result / self._dx ** 2

        bulk_energy = self._dEdp(rho)  # shape: (N_species, Nx, Ny)

        lap_rho = laplacian(rho)
        chemical_potential = bulk_energy - (self._interface_scalar * self._k_laplacian * lap_rho)
        lap_d_rho = laplacian(chemical_potential)

        return rho + self._M * lap_d_rho * self._dt

    @partial(jax.jit, static_argnums=(0, ))
    def _evolve_allen_cahn(self, phi):
        """ Explicit Euler step for Allen-Cahn """
        local_rhos_per_bin = self.get_local_rho_species(phi, self.bin_indices)  # Shape: (N_bins, N_species)

        def bulk_term_for_bin(local_rho):
            return jax.vmap(self._dEdp, in_axes=(0, None))(
                jnp.arange(self._model.N_species()), local_rho
            )

        bulk_term_scalar = jax.vmap(bulk_term_for_bin)(local_rhos_per_bin)
        bulk_energy = bulk_term_scalar.reshape((self._model.N_species(),) + phi.shape[1:])

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
