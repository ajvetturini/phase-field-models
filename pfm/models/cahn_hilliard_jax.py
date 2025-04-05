"""
Cahn-Hilliard implementation
"""
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)

class jCahnHilliard:
    def __init__(self, free_energy_model, config, integrator, rng):
        self._rng = rng
        self.N = config.get('N')
        self.k_laplacian = config.get('k', 1.0)
        self.dt = config.get('dt')
        self.M = config.get('M', 1.0)
        self.dx = config.get('dx', 1.0)
        self._internal_to_user = config.get('distance_scaling_factor', 1.0)
        self._user_to_internal = 1.0 / self._internal_to_user
        self.dim = config.get('dim', 2)
        self.free_energy_model = free_energy_model
        if self.dim <= 0 or self.dim > 2:
            raise Exception('Unable to proceed, currently only support for 1D and 2D is implemented')

        log2N = jnp.log2(self.N)
        if jnp.mod(self.N, 2) != 0.:
            raise ValueError("N should be a power of 2")

        self.N_minus_one = self.N - 1
        self.bits = int(log2N)

        self.grid_size = self.N ** self.dim
        self._grid_size_str = "x".join([str(self.N)] * self.dim)

        num_species = free_energy_model.N_species()
        shape = tuple([num_species] + [self.N] * self.dim)
        rho = np.zeros(shape)

        # Setup RNG key for initialization
        prng = jax.random.PRNGKey(rng)   # rng is just a seed in this case

        if "load_from" in config:
            filename = config.get('load_from')
            with open(filename, 'r') as load_from:
                for s in range(num_species):
                    if self.dim == 1:
                        for idx in range(self.N):
                            rho[s, idx] = float(load_from.readline().strip())
                    elif self.dim == 2:
                        coords = [0, 0]
                        for coords[1] in range(self.N):
                            for coords[0] in range(self.N):
                                idx = self.cell_idx(coords)
                                rho[s, coords[1], coords[0]] = float(load_from.readline().strip())
                    else:
                        raise ValueError(f"Unsupported number of dimensions {self.dim}")
        elif "initial_density" in config:
            initial_density = config.get('initial_density')
            densities = [float(initial_density)] * num_species
            initial_A = config.get('initial_A', 1e-2)
            initial_N_peaks = config.get('initial_N_peaks', 0)
            initial_k = 2 * np.pi * initial_N_peaks / self.N  # Wave vector of modulation
            for bin in range(self.grid_size):
                modulation = initial_A * np.cos(initial_k * bin)
                for i in range(num_species):
                    prng, k1 = jax.random.split(prng, 2)
                    rand1 = jax.random.uniform(k1, minval=0.0, maxval=1.0)
                    random_factor = (rand1 - 0.5 if initial_N_peaks == 0 else
                                     1.0 + 0.02 * (rand1 - 0.5))
                    average_rho = densities[i]
                    coords = self.fill_coords(bin)
                    if self.dim == 1:
                        rho[i, bin] = average_rho * (1.0 + 2.0 * modulation * random_factor)
                    elif self.dim == 2:
                        rho[i, coords[1], coords[0]] = average_rho * (1.0 + 2.0 * modulation * random_factor)
                    elif self.dim == 3:
                        raise Exception('Invalid dimension specified, only supports 1D and 2D.')
        else:
            raise ValueError("Either 'initial_density' or 'load_from' should be specified")

        self.dx *= self._user_to_internal  # Proportional to M
        self.M /= self._user_to_internal  # Proportional to M^-1
        self.k_laplacian *= self._user_to_internal ** 5  # Proportional to M^5
        rho /= self._user_to_internal ** 3

        self.V_bin = self.dx ** 3

        self.integrator = integrator
        self._output_ready = False
        self._d_vec_size = 0
        self._grid_size_str = ""
        dtype = config.get('float_type', jnp.float64)
        self._float_type = dtype
        self.init_rho = jnp.array(rho, dtype=dtype)  # initialized rho


    def fill_coords(self, idx):
        coords = []
        for d in range(self.dim):
            coords.append(idx & self.N_minus_one)
            idx >>= self.bits
        return coords

    def cell_idx(self, coords):
        idx = 0
        multiply_by = 1
        for d in range(self.dim):
            idx += coords[d] * multiply_by
            multiply_by <<= self.bits
        return idx

    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, field: jnp.ndarray, species: int, coords: tuple) -> jnp.ndarray:
        grad = []

        for d in range(self.dim):
            coords_plus = list(coords)
            coords_minus = list(coords)

            coords_plus[d] = (coords[d] + 1) % self.grid_size
            coords_minus[d] = (coords[d] - 1 + self.grid_size) % self.grid_size

            idx_plus = (species, *coords_plus)
            idx_minus = (species, *coords_minus)

            diff = (field[idx_plus] - field[idx_minus]) / (2 * self.dx)
            grad.append(diff)

        return jnp.array(grad, dtype=self._float_type)

    def evolve(self, rho):
        return self.integrator.evolve(rho)

    @partial(jax.jit, static_argnums=(0,))
    def average_mass(self, rho: jnp.array):
        """
        Calculates the average mass density, i.e., mass per unit grid volume averaged across all bins and all species
        rho: jnp.array of shape (N_species, **dim) (e.g., (N_species, Nx), (N_species, Nx, Ny), or
             (N_species, Nx, Ny, Nz)

        This will return a single floating point value
        """
        mass_density = jnp.sum(rho, axis=0)  # sum over species
        total_mass = jnp.sum(mass_density) * self.V_bin
        total_volume = self.V_bin * jnp.prod(jnp.array(rho.shape[1:]))
        return total_mass / total_volume

    @partial(jax.jit, static_argnums=(0,))
    def mass_per_species(self, rho: jnp.ndarray,):
        return jnp.sum(rho, axis=tuple(range(1, rho.ndim))) * self.V_bin

    @partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the average free energy per bin, including bulk and interfacial contributions.
        Assumes: rho.shape = (N_species, Nx, Ny) or similar
        """
        num_bins = jnp.prod(jnp.array(rho.shape[1:], dtype=int))

        def bin_free_energy(i, acc):
            coords = jnp.unravel_index(i, rho.shape[1:])
            interfacial_contrib = 0.0

            def species_loop(species_idx, interf_acc):
                grad = self.gradient(rho, species_idx, coords)
                return interf_acc + self.k_laplacian * jnp.sum(grad ** 2)

            interfacial_contrib = jax.lax.fori_loop(0, rho.shape[0], species_loop, 0.0)

            rho_species_at_coords = rho[(slice(None),) + coords]
            bulk = self.free_energy_model.bulk_free_energy(rho_species_at_coords)

            return acc + bulk + interfacial_contrib

        total_fe = jax.lax.fori_loop(0, num_bins, bin_free_energy, 0.0)
        avg_fe = total_fe * self.V_bin / num_bins
        return avg_fe

    def print_species_density(self, species, output, t, rho):
        output.write(f"# step = {t}, t = {t * self.dt:.5f}, size = {self._grid_size_str}\n")
        rho = np.array(rho)  # Convert to numpy array for easier writing
        for idx in range(np.prod(rho.shape[1:])):
            coords = self.fill_coords(idx)
            if idx > 0:
                modulo = self.N
                for d in range(1, self.dim):
                    if idx % modulo == 0:
                        output.write("\n")
                    modulo <<= self.bits
            if self.dim == 1:
                output.write(f"{self._density_to_user(rho[species, coords[0]])} ")
            else:
                output.write(f"{self._density_to_user(rho[species, coords[1], coords[0]])} ")
        output.write("\n")

    def _density_to_user(self, v):
        return v / (self._internal_to_user ** 3)
