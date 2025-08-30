import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from copy import deepcopy


class PhaseFieldModel:
    def __init__(self, free_energy_model, config, integrator, rng, field_name, **kwargs):
        self._rng = rng
        self.N = config.get('N')
        self.dt = config.get('dt')
        self.dx = config.get('dx', 1.0)
        self._distance_scaling_factor = config.get('distance_scaling_factor', 1.0)
        self._inverse_scaling_factor = 1.0 / self._distance_scaling_factor
        self.dim = config.get('dim', 2)
        self.free_energy_model = free_energy_model
        self.field_name = field_name  # Name of the order parameter (e.g., rho or phi)

        if self.dim <= 0 or self.dim > 3:
            raise Exception('Unable to proceed, currently only support for 1D, 2D, 3D is implemented')

        if np.mod(self.N, 2) != 0:
            raise ValueError("N should be a power of 2")

        num_species = free_energy_model.N_species()
        shape = tuple([num_species] + [self.N] * self.dim)
        initial_field = np.zeros(shape)

        # Setup RNG key for initialization
        prng = jax.random.PRNGKey(int(rng))   # rng is just a seed in this case

        # Load in the initial state which we handle here using either a specified state (load_from)
        # or from a procedure sefined in https://github.com/lorenzo-rovigatti/cahn-hilliard
        # Otherwise, if the user specifies "custom_function" then the custom_fn is used to init the initial_field
        if "load_from" in config:
            filename = config.get('load_from')
            with open(filename, 'r') as load_from:
                first_line = load_from.readline().strip()  # Skip the first line which is a commented line (#)
                if first_line.startswith('#'):
                    pass  # Skip the commented line
                else:
                    # If the first line is not a comment, process it
                    # Assuming the first line contains data for the first species (s=0)
                    if self.dim == 1:
                        values = first_line.split()
                        for idx, val_str in enumerate(values):
                            if idx < self.N:
                                initial_field[0, idx] = float(val_str)
                    elif self.dim == 2:
                        values = first_line.split()
                        # Assuming the first line contains N*self.dim values for the first species
                        for i in range(self.N):
                            start_index = i * self.dim
                            end_index = (i + 1) * self.dim
                            if end_index <= len(values):
                                initial_field[0, i, :] = [float(v) for v in values[start_index:end_index]]

                    elif self.dim == 3:
                        raise Exception('3D not fully supported yet.')

                for s in range(num_species):
                    for ct, line in enumerate(load_from):
                        line = line.strip()
                        if line:  # Ensure the line is not empty
                            values = line.split()
                            if self.dim == 1:
                                for idx, val_str in enumerate(values):
                                    if idx < self.N:
                                        initial_field[s, idx] = float(val_str)
                                break  # Move to the next species after reading enough values
                            elif self.dim == 2:
                                # Assuming each line contains N * self.dim values for one species
                                if len(values) == self.N:
                                    initial_field[s, ct, :] = [float(v) for v in values]
                                else:
                                    raise Exception('Invalid load_from file')

                            elif self.dim == 3:
                                raise Exception('Unable to proceed, 3D not fully implemented yet.')
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

                initial_field = densities[:, None] * (1.0 + 2.0 * modulation[None, :] * random_factor)

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

                initial_field = densities[:, None, None] * (1.0 + 2.0 * modulation[None, :, :] * random_factor)

            elif self.dim == 3:
                x, y, z = jnp.meshgrid(
                    jnp.arange(self.N), jnp.arange(self.N), jnp.arange(self.N)
                )
                r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
                modulation = initial_A * jnp.cos(k * r)
                prng, subkey = jax.random.split(prng)
                noise = jax.random.uniform(subkey, shape=(num_species, self.N, self.N, self.N))

                if initial_N_peaks == 0:
                    random_factor = noise - 0.5
                else:
                    random_factor = 1.0 + 0.02 * (noise - 0.5)

                initial_field = densities[:, None, None, None] * (
                        1.0 + 2.0 * modulation[None, :, :, :] * random_factor
                )

        elif "custom_initial_condition" in config:  # User must specify in the TOML custom_initial_condition = true
            fn = kwargs.get('custom_fn')
            ishape = deepcopy(initial_field.shape)
            initial_field = fn(initial_field)  # Use the init array of correct size
            assert np.array_equal(np.array(ishape), np.array(initial_field.shape)), 'ERROR: Changed shape during init'

        else:
            raise ValueError("Either 'initial_density', 'custom_initial_condition', or 'load_from' should be specified")

        self.dx *= self._inverse_scaling_factor
        initial_field /= self._inverse_scaling_factor ** 3
        self.V_bin = self.dx ** 3

        self.integrator = integrator
        dtype = config.get('float_type', jnp.float32)
        self._float_type = dtype
        if dtype == jnp.float64:
            print('NOTE: 64-bit precision specified, operations will be slow if not on double precision specific '
                  'hardware. Even then, the roll laplacian is quite slow.')
        setattr(self, f"init_{self.field_name}", jnp.array(initial_field, dtype=dtype))

        # Central differences can be "bad" for explicit euler and can blow up the energy calculations, so
        # forward difference is used for stability by default (but can be switched)
        # Should look into this a bit more when time arieses
        self._grad_diff_method = config.get('pfm_diff_method', 'fwd')

    def get_initial_condition(self):
        """
        Returns the initial condition of the order parameter field.
        """
        return getattr(self, f"init_{self.field_name}")

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

    @partial(jax.jit, static_argnums=(0,))
    def average_mass(self, rho: jnp.array):
        """
        Calculates the average mass density, i.e., mass per unit grid volume averaged across all bins and all species

        This will return a single floating point value
        """
        mass_density = jnp.sum(rho, axis=0)  # sum over species
        total_mass = jnp.sum(mass_density) * self.V_bin
        total_volume = self.V_bin * jnp.prod(jnp.array(rho.shape[1:]))
        return total_mass / total_volume

    def print_species_density(self, species, output, t, rho):
        """ Prints the density of a given species to the output file  """
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
        return v / (self._distance_scaling_factor ** 3)

    def evolve(self, field):
        raise NotImplementedError("Evolve method must be implemented in the derived class")

    def average_free_energy(self, field):
        raise NotImplementedError('Average free energy must be implemented in the derived class')
