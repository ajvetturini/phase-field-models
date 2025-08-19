# Semi-Implicit method, only really useful for Cahn-Hilliard because of 4th order dependency
# Note that this is a FFT-based implicit, thus requires PERIODIC bcs

import jax.numpy as jnp
from pfm.integrators.base_integrator import Integrator
import jax
from functools import partial

class SemiImplicitSpectral(Integrator):
    def __init__(self, model, config):
        super().__init__(model, config)
        self._dx = jnp.array(self.dx, dtype=self._float_type)
        self._dt = jnp.array(self._dt, dtype=self._float_type)

        self.ndim = self._dim  # grid_shape = (Nx,) or (Nx,Ny) or (Nx,Ny,Nz)


        # self._dim in {1,2,3}; self._N_per_dim is int or tuple
        if isinstance(self._N_per_dim, int):
            sizes = (self._N_per_dim,) * self._dim
        else:
            assert len(self._N_per_dim) == self._dim
            sizes = tuple(self._N_per_dim)

        # Build Fourier k^2, k^4 operators
        k_axes = [jnp.fft.fftfreq(n, d=self._dx) * 2 * jnp.pi for n in sizes]
        mesh = jnp.meshgrid(*k_axes, indexing="ij")
        k2 = sum(kk ** 2 for kk in mesh)  # shape: sizes
        self._k2 = k2[None, ...]  # shape: (1, *sizes)
        self._k4 = (k2 ** 2)[None, ...]  # shape: (1, *sizes)

        if config.get('verbose', True):
            print(f"Semi-Implicit Spectral Integrator ({self.ndim}D)")
            if self._use_autodiff:
                print("Using autodiff for bulk energy derivative")

        # In the init we set the evolve function, allowing us to easily change what data gets transferred to individual
        # energy models:
        if config.get('model', 'ch').lower() == 'ch':
            self._evolve_fn = self._evolve_cahn_hilliard
        elif config.get('model', 'ac').lower() == 'ac':
            self._evolve_fn = self._evolve_allen_cahn
        else:
            raise Exception('Invalid integrator method, valid options are: `ch`, `ac`, ')

    @partial(jax.jit, static_argnums=(0,))
    def _evolve_cahn_hilliard(self, rho):
        """
        Semi-implicit spectral update for Cahn–Hilliard:
        ρ^{n+1}(k) = (ρ^n(k) - dt * M k^2 * FFT[f'(ρ^n)]) / (1 + dt * M κ k^4)
        """
        kappa = self._interface_scalar * self._k_laplacian

        # Bulk derivative f'(rho) in real space
        bulk_energy = self._dEdp(rho)  # shape: (N_species, ...)

        axes = tuple(range(1, rho.ndim))  # transform over spatial axes
        rho_hat = jnp.fft.fftn(rho, axes=axes)
        bulk_hat = jnp.fft.fftn(bulk_energy, axes=axes)

        # Ensure k-tensors broadcast over species
        k2 = self._k2  # shape: (1, ...)
        k4 = self._k4  # shape: (1, ...)

        numerator = rho_hat - self._dt * self._M * k2 * bulk_hat
        denominator = 1.0 + self._dt * self._M * kappa * k4
        rho_hat_new = numerator / denominator
        rho_new = jnp.fft.ifftn(rho_hat_new, axes=axes).real
        return rho_new

    @partial(jax.jit, static_argnums=(0,))
    def _evolve_allen_cahn(self, phi):
        """
        Semi-implicit spectral update for Allen–Cahn:
        φ^{n+1}(k) = (φ^n(k) - dt * L * FFT[f'(φ^n)]) / (1 + dt * L κ k^2)
        """
        bulk_energy = self._dEdp(phi)  # f'(phi) in real space

        phi_hat = jnp.fft.fftn(phi, axes=tuple(range(1, phi.ndim)))
        bulk_hat = jnp.fft.fftn(bulk_energy, axes=tuple(range(1, bulk_energy.ndim)))

        numerator = phi_hat - self._dt * self._L_phi * bulk_hat
        denominator = 1.0 + self._dt * self._L_phi * self._k_laplacian * self._k2
        phi_hat_new = numerator / denominator

        phi_new = jnp.fft.ifftn(phi_hat_new, axes=tuple(range(1, phi.ndim))).real
        return phi_new

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

