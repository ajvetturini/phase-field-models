from pfm import SimulationManager
import toml
import time
import numpy as np
from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
from scipy.ndimage import gaussian_filter
jax.config.update("jax_enable_x64", True)

class GrainGrowth(FreeEnergyModel):
    def __init__(self, config):
        super().__init__(config)
        self._energy_config = config.get('grain_growth')
        self._N_phases = self._energy_config.get('N_phases')
        self._W = self._energy_config.get('W')
        self._gamma = self._energy_config.get('gamma')

    def N_species(self):
        return self._N_phases

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, phi):
        """ Derivative of the double-well bulk free energy. """
        phi_i = phi[species]  # Phi is of the shape (N_species, Nx, Ny)

        # Term 1: Derivative of W * phi_i^2 * (1 - phi_i)^2
        # 2 * W * phi_i * (2 * phi_i - 1) * (phi_i - 1)
        term1 = 2.0 * self._W * phi_i * (2.0 * phi_i - 1.0) * (phi_i - 1.0)

        # Term 2: Derivative of gamma * sum_{k<j} phi_k^2 * phi_j^2
        # Needs sum_{j != i} phi_j^2
        # Compute sum(phi_k^2 for all k)
        sum_phi_sq_all = jnp.sum(jnp.square(phi))
        # Subtract phi_i^2 to get sum(phi_j^2 for j != i)
        sum_phi_sq_others = sum_phi_sq_all - jnp.square(phi_i)

        # 2 * gamma * phi_i * sum_{j != i} phi_j^2
        term2 = 2.0 * self._gamma * phi_i * sum_phi_sq_others

        return term1 + term2

    def bulk_free_energy(self, phi):
        term1_sum = jnp.sum(jnp.square(phi) * jnp.square(1.0 - phi))

        # Compute sum_{i<j} phi_i^2 * phi_j^2 efficiently
        # (sum_i phi_i^2)^2 = sum_i phi_i^4 + 2 * sum_{i<j} phi_i^2 * phi_j^2
        # sum_{i<j} phi_i^2 * phi_j^2 = 0.5 * [ (sum_i phi_i^2)^2 - sum_i phi_i^4 ]
        phi_sq = jnp.square(phi)
        sum_phi_sq = jnp.sum(phi_sq)
        sum_phi_fourth = jnp.sum(jnp.square(phi_sq))  # sum phi_i^4
        term2_sum = 0.5 * (jnp.square(sum_phi_sq) - sum_phi_fourth)

        f_bulk = self._W * term1_sum + self._gamma * term2_sum
        return f_bulk


def custom_initial_condition(init_phi, n_grain_seeds=None, seed=8, sigma=1.5, noise_amplitude=0.05):
    """
    Generates an initial condition with random grains for multi-phase grain growth.
    Adds smoothing and small random noise to enable nontrivial evolution.

    Args:
        N_phases (int): Number of grain orientations.
        Nx, Ny (int): Grid dimensions.
        n_grain_seeds (int or None): Number of initial seeds. Defaults to N_phases.
        seed (int): RNG seed.
        sigma (float): Std. dev. for Gaussian smoothing.
        noise_amplitude (float): Max amplitude of noise added to each phi_i.
    Returns:
        np.ndarray: Array of shape (N_phases, Nx, Ny)
    """
    N_phases, Nx, Ny = init_phi.shape
    rng = np.random.default_rng(seed)
    phi = np.zeros((N_phases, Nx, Ny), dtype=np.float32)

    if n_grain_seeds is None:
        n_grain_seeds = N_phases

    # Random seed positions and orientations
    seed_x = rng.integers(0, Nx, size=n_grain_seeds)
    seed_y = rng.integers(0, Ny, size=n_grain_seeds)
    seed_ids = rng.integers(0, N_phases, size=n_grain_seeds)

    # Assign grid points to nearest seed
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    for i in range(n_grain_seeds):
        dx = X - seed_x[i]
        dy = Y - seed_y[i]
        dist_sq = dx**2 + dy**2

        if i == 0:
            min_dist_sq = dist_sq.copy()
            orientation_map = np.full((Nx, Ny), seed_ids[i], dtype=int)
        else:
            mask = dist_sq < min_dist_sq
            orientation_map[mask] = seed_ids[i]
            min_dist_sq[mask] = dist_sq[mask]

    # One-hot encode
    for i in range(N_phases):
        phi[i, :, :] = (orientation_map == i).astype(np.float32)

    # Smooth each channel
    for i in range(N_phases):
        phi[i] = gaussian_filter(phi[i], sigma=sigma)

    # Normalize so sum_i phi_i = 1
    phi_sum = np.sum(phi, axis=0, keepdims=True)
    phi /= (phi_sum + 1e-8)  # Avoid division by zero

    # Add small random noise and renormalize
    noise = rng.uniform(-noise_amplitude, noise_amplitude, size=phi.shape).astype(np.float32)
    phi += noise
    phi = np.clip(phi, 0.0, 1.0)

    phi_sum = np.sum(phi, axis=0, keepdims=True)
    phi /= (phi_sum + 1e-8)

    return phi


c = toml.load('grain_growth_input.toml')
manager = SimulationManager(c, custom_energy=GrainGrowth, custom_initial_condition=custom_initial_condition)
start = time.time()
manager.run()
end = time.time() - start

minutes = int(end // 60)
seconds = int(end % 60)

print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')