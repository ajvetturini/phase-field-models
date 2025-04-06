"""
Allen-Cahn implementation
"""
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)

class AllenCahn:
    def __init__(self, free_energy_model, config, integrator, rng):
        self._rng = rng
        self.N = config.get('N')
        self.gamma = config.get('gamma', 1.0)  # Mobility/kinetic coefficient
        self.dt = config.get('dt')
        self.dx = config.get('dx', 1.0)
        self._k = config.get('k', 1.0)
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

        num_species = free_energy_model.N_species()
        shape = tuple([num_species] + [self.N] * self.dim)
        phi = np.zeros(shape)

        # Setup RNG key for initialization
        prng = jax.random.PRNGKey(rng)   # rng is just a seed in this case

        if "load_from" in config:
            filename = config.get('load_from')
            with open(filename, 'r') as load_from:
                for s in range(num_species):
                    if self.dim == 1:
                        for idx in range(self.N):
                            phi[s, idx] = float(load_from.readline().strip())
                    elif self.dim == 2:
                        for i in range(self.N):
                            for j in range(self.N):
                                phi[s, i, j] = float(load_from.readline().strip())
                    else:
                        raise ValueError(f"Unsupported number of dimensions {self.dim}")
        elif "initial_density" in config:
            initial_density = config.get('initial_density')
            densities = np.array([float(initial_density)] * num_species)
            initial_A = config.get('initial_A', 1e-2)
            initial_N_peaks = config.get('initial_N_peaks', 0)
            k = 2 * np.pi * initial_N_peaks / self.N  # Wave vector of modulation
            if self.dim == 1:
                x = jnp.arange(self.N)  # shape: (Nx,)
                modulation = initial_A * jnp.cos(k * x)  # shape: (Nx,)
                prng, subkey = jax.random.split(prng)
                noise = jax.random.uniform(subkey, shape=(num_species, self.N))  # shape: (S, Nx)

                if initial_N_peaks == 0:
                    random_factor = noise - 0.5
                else:
                    random_factor = 1.0 + 0.02 * (noise - 0.5)

                phi = densities[:, None] * (1.0 + 2.0 * modulation[None, :] * random_factor)

            elif self.dim == 2:
                x, y = jnp.meshgrid(jnp.arange(self.N), jnp.arange(self.N))
                r = jnp.sqrt(x ** 2 + y ** 2)
                modulation = initial_A * jnp.cos(k * r)
                prng, subkey = jax.random.split(prng)
                noise = jax.random.uniform(subkey, shape=(num_species, self.N, self.N))

                if initial_N_peaks == 0:
                    random_factor = noise - 0.5
                else:
                    random_factor = 1.0 + 0.02 * (noise - 0.5)

                phi = densities[:, None, None] * (1.0 + 2.0 * modulation[None, :, :] * random_factor)

            elif self.dim == 3:
                raise NotImplementedError("3D initialization not implemented.")
        else:
            raise ValueError("Either 'initial_density' or 'load_from' should be specified")

        self.dx *= self._user_to_internal  # Proportional to M
        self.gamma *= self._user_to_internal ** 3 # Scaling for mobility/kinetic coefficient
        phi /= self._user_to_internal ** 3

        self.V_bin = self.dx ** 3

        self.integrator = integrator
        self._output_ready = False
        self._grid_size_str = ""
        dtype = config.get('float_type', jnp.float64)
        self._float_type = dtype
        if dtype != jnp.float64:
            print('NOTE: 64-bit precision not being used, stability may be off.')
        self.init_phi = jnp.array(phi, dtype=dtype)  # initialized phi

        self._grad_diff_method = config.get('ac_diff_method', 'fwd')  # Forward or central difference in gradient

    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, field: jnp.ndarray) -> jnp.ndarray:
        grads = []
        spatial_axes = tuple(range(1, field.ndim))

        for d in range(self.dim):
            axis_to_roll = spatial_axes[d]
            if self._grad_diff_method.lower() == 'fwd':
                grad_d = (jnp.roll(field, -1, axis=axis_to_roll) - field) / self.dx
            elif self._grad_diff_method.lower() == 'central':
                grad_d = (jnp.roll(field, -1, axis=axis_to_roll) - jnp.roll(field, 1, axis=axis_to_roll)
                          ) / (2 * self.dx)
            else:
                raise Exception(f'Unsupported difference method: {self._grad_diff_method}. '
                                f'Valid options are "fwd" and "central".')
            grads.append(grad_d)
        return jnp.stack(grads, axis=1)

    def evolve(self, rho):
        return self.integrator.evolve(rho, method='ac')

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
    def average_free_energy(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the average free energy per bin for the Allen-Cahn model.
        Assumes: phi.shape = (N_species, Nx, Ny) or similar
        """
        num_bins = jnp.prod(jnp.array(phi.shape[1:], dtype=jnp.int32))

        # 1. Calculate gradient contribution
        all_gradients = self.gradient(phi)
        sum_sq_grad = jnp.sum(all_gradients ** 2, axis=1)
        interfacial_density = 0.5 * self._k * jnp.sum(sum_sq_grad, axis=0)

        # 2. Calculate bulk contribution
        bulk_density = self.free_energy_model.bulk_free_energy(phi)

        # 3. Combine and average
        total_free_energy_density = bulk_density + interfacial_density
        total_fe = jnp.sum(total_free_energy_density) * self.V_bin
        avg_fe = total_fe / num_bins

        return avg_fe

    def print_species_density(self, species, output, t, rho):
        """
        Prints the density of a given species to the output file in a more Pythonic/NumPy way.

        Args:
            species (int): The index of the species to print.
            output (file object): The file to write the density to.
            t (int): The current time step.
            rho (jnp.array): The density array with shape (N_species, *self._N_per_dim).
        """
        size_str = "x".join([str(self.N)] * self.dim)
        output.write(f"# step = {t}, t = {t * self.dt:.5f}, size = {size_str}\n")
        rho_np = np.array(rho[species])  # Get the density of the specified species as a NumPy array

        if self.dim == 1:
            output.write(" ".join(f"{self._density_to_user(val)}" for val in rho_np))
        elif self.dim == 2:
            for row in rho_np:
                output.write(" ".join(f"{self._density_to_user(val)}" for val in row))
                output.write("\n")
        else:
            raise NotImplementedError(f"Printing for {self.dim} dimensions is not implemented.")

        output.write("\n")

    def _density_to_user(self, v):
        return v / (self._internal_to_user ** 3)
