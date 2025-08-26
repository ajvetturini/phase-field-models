import numpy as np
import toml
import jax
import jax.numpy as jnp
from flax import linen as nn
from pfm.energy_models import Landau, SimpleWertheim, GenericWertheim, SalehWertheim
from pfm.integrators import ExplicitEuler, SemiImplicitSpectral
from pfm.pinn import MLP, train_ch, train_ac
from pfm.phase_field_models import CahnHilliard, AllenCahn
import os
from functools import partial
import time


class SimulationManager:
    def __init__(self, config, custom_energy=None, custom_initial_condition=None):
        self._steps = int(config.get('steps', 100))
        self._print_mass_every = int(config.get('print_every', 10))
        self._print_trajectory_strategy = config.get('print_trajectory_strategy', 'linear').lower()
        if self._print_trajectory_strategy == 'linear':
            self._print_trajectory_every = int(config.get("print_trajectory_every"))  # Must be specified to prevent
            # accidental massive writes
            self._print_last_every = config.get('print_last_every', self._print_trajectory_every)

        elif self._print_trajectory_strategy == 'log':
            self._log_n0 = config.get('log_n0')
            self.log_fact = config.get('log_fact')
            self._print_last_every = config.get('print_last_every')
        else:
            raise Exception('Invalid printing strategy, valid options are linear and log.')

        self._config = config
        self._write_path = config.get('write_path', r'./')  # Assumes same directory writing by default

        # Set RNG:
        self._rng_seed = int(config.get('steps', np.random.randint(1000000)))
        if 'steps' not in config:
            print(f'RNG seed not specified, using seed: {self._rng_seed}')

        # Setup the free energy model, integrator, and system based on if jax is being used:
        # The custom_energy and custom_initial_conditions might be None (which is fine) but are still passed in
        # The use of the TOML config dictates the use of these custom functions
        self._free_energy_model = _read_in_energy_model(config, config.get('free_energy'), custom_energy)
        self._integrator = self._read_in_integrator(self._free_energy_model, config, config.get('integrator', 'euler'))
        self._system = _read_in_model(self._free_energy_model, config, config.get('model', 'ch'),
                                           self._integrator, self._rng_seed, custom_fn=custom_initial_condition)
        self._traj_printed = 0
        self._trajectories = []
        self._custom_initial_condition = custom_initial_condition

        # Store init data from system:
        name = 'init_' + self._system.field_name
        self.init_field = getattr(self._system, name)

    def close(self):
        for traj in self._trajectories:
            if traj:
                traj.close()

    @staticmethod
    def _read_in_integrator(model, config, integrator_name):
        if integrator_name.lower() == 'euler':
            return ExplicitEuler(model, config)
        elif integrator_name.lower() == 'semi_implicit' or integrator_name.lower() == 'spectral':
            return SemiImplicitSpectral(model, config)
        else:
            raise Exception('Invalid integrator scheme, valid options are: euler, semi_implicit (or spectral), ')

    def _print_current_state(self, prefix, t, rho=None):
        print(f"{prefix} state at time {t}")
        num_species = self._free_energy_model.N_species()
        for i in range(num_species):
            filename = f"{prefix}{i}.dat"
            fp = os.path.join(self._write_path, filename)
            with open(fp, "w") as output:
                self._system.print_species_density(i, output, t, rho)

    def _should_print_last(self, t):
        return self._print_last_every > 0 and t % self._print_last_every == 0

    def _should_print_traj(self, t):
        if self._print_trajectory_strategy == "linear":
            return self._print_trajectory_every > 0 and t % self._print_trajectory_every == 0

        elif self._print_trajectory_strategy.lower() == "log":
            if not hasattr(self, '_log_n0'):
                return False
            next_t = int(round((self._log_n0 * (self.log_fact ** self._traj_printed))))
            return next_t == t

        return False

    def average_free_energy(self, rho=None):
        """ Cahn-Hilliard (or Allen-Cahn) will simply calculate this based on stored rho values """
        return self._system.average_free_energy(rho)

    def average_mass(self, rho=None):
        """ Cahn-Hilliard (or Allen-Cahn) will simply calculate this based on stored rho values """
        return self._system.average_mass(rho)

    """
    Main run method
    """

    def _run_cpu(self):
        """ Main function of the Simulation Manager, runs the simulation. This is a "slow run" which does not leverage
         jax outside of jit-ting the process. This should result in similar to CPU based C++ implementation
         (but only for flaot32), but is not well-suited for very large (1e9-1e10) timestep simulations
         """
        if self._steps > 1e6:
            print('WARNING: CPU based phase field model will run quite slowly!')
        # Setup simulation trajectory tracking:
        num_species = self._free_energy_model.N_species()
        if self._print_trajectory_every > 0 or hasattr(self, '_log_n0'):  # Check if log printing is enabled
            for i in range(num_species):
                def_name = os.path.join(self._write_path, f"trajectory_{i}.dat")
                self._trajectories.append(open(def_name, "w"))

        name = 'init_' + self._system.field_name
        rho_0 = getattr(self._system, name)  # Error will raise if not specified

        if self._config.get('verbose', True):
            self._print_current_state("init_", 0, rho=rho_0)
        fp = os.path.join(self._write_path, 'energy.dat')
        rho_n = rho_0  # init
        with open(fp, "w") as mass_output:
            for t in range(self._steps):
                if self._should_print_last(t) and self._config.get('verbose', True):
                    self._print_current_state("most_recent_", t, rho=rho_n)

                if self._should_print_traj(t) and self._config.get('verbose', True):
                    num_species = self._free_energy_model.N_species()
                    for i in range(num_species):
                        self._system.print_species_density(i, self._trajectories[i], t, rho_n)
                    self._traj_printed += 1

                # This is the write out to the trajectory
                if self._print_mass_every > 0 and t % self._print_mass_every == 0:
                    output_line = (f"{t * self._system.dt:.5f} {self.average_free_energy(rho_n):.8f} "
                                   f"{self.average_mass(rho_n):.5f} {t}")
                    mass_output.write(output_line + "\n")
                    if self._config.get('verbose', True):
                        print(output_line)

                rho_n = self._system.evolve(rho_n)

        if self._config.get('verbose', True):
            # Print the final state:
            self._print_current_state("last_", self._steps, rho=rho_n)

        self.close()  # Close out write files (although automatic garbage collection should do this)

    def _run_jax(self):
        """ uses evolve_n_steps and proper logging in arrays to store information """
        name = 'init_' + self._system.field_name
        rho_n = getattr(self._system, name)  # Error will raise if not specified
        self._print_current_state("init_", 0, rho=rho_n)  # Print init state

        # Simulation dimension string:
        N, Ns, dim = self._system.N, self._free_energy_model.N_species(), self._system.dim
        if dim == 1:
            dim_str = f'{N}'
        elif dim == 2:
            dim_str = f'{N}x{N}'
        elif dim == 3:
            dim_str = f'{N}x{N}x{N}'
        else:
            raise Exception('Invalid value of dimension')

        # Setup logs:
        energy_log = [self._log_energy(0, rho_n)]
        traj_log = [np.array(rho_n, dtype=np.float32)]
        steps = [0]

        # Logging intervals:
        log_mass_every = self._print_mass_every
        log_traj_every = self._print_trajectory_every
        log_every = np.gcd(log_mass_every, log_traj_every)

        # Begin simulation
        current_step = 0
        while current_step < self._steps:
            # Determine how many steps to take in the next block
            if self._config.get('verbose', True):
                print(f'Beginning step {current_step}')
            remaining_steps = self._steps - current_step
            num_steps_to_run = min(log_every, remaining_steps)

            # Advance the simulation
            rho_n = self._evolve_n_steps(rho_n, num_steps_to_run)
            current_step += num_steps_to_run

            # Conditional logging
            if current_step % log_mass_every == 0:
                energy_log.append(self._log_energy(current_step, rho_n))
            if current_step % log_traj_every == 0:
                traj_log.append(np.array(rho_n, dtype=np.float32))
                steps.append(current_step)

        # Print final state and logs
        energy_log = np.array(energy_log)
        traj_log = np.array(traj_log)
        self._print_current_state("last_", self._steps, rho=rho_n)

        print('Beginning write out of final files, this may take a moment...')
        self._write_output_jax_arrays(traj_log, energy_log, steps, dim_str, rho_n)

    def run(self, override_use_jax: bool = False):
        """ Top level method to run a simulation """
        devices = jax.devices()
        accelerator_found = False
        for d in devices:
            if d.platform.lower() == 'gpu' or d.platform.lower() == 'tpu':
                accelerator_found = True
                print('Accelerator found')
                break
            elif d.platform.lower() not in ['gpu', 'tpu', 'cpu']:
                raise Exception(f'Unknown hardware device: {d.platform}')

        t = time.time()
        if accelerator_found or override_use_jax:
            self._run_jax()
        else:
            # This still leverages JIT / JAX w/ performance similar to C++, but write out happens a bit more smoothly
            self._run_cpu()
        end = time.time()
        return end - t

    def _write_output_jax_arrays(self, traj_log, energy_log, steps, dim_string, final_rho):
        """ Writes a simple ASCII file of the trajectory and energies tracked during the simulation """
        shaped_energy = energy_log
        write_list = []
        s1, s2 = 0, 0
        for i in shaped_energy:
            write_list.append(f'{s1:.5f} {i[1]:.8f} {i[2]:.5f} {s2:d}')
            if s1 == 0:
                add1 = shaped_energy[1][0]
                add2 = int(shaped_energy[1][-1])

                if add1 == 0:
                    add1 = 1
                if add2 == 0:
                    add2 = 1
            s1 += add1
            s2 += add2

        with open(os.path.join(self._write_path, 'energy.dat'), 'w') as file:
            for item in write_list:
                file.write(item + '\n')

        # Repeat for trajectory:
        n_traj, n_spec = traj_log.shape[0], traj_log.shape[1]
        for i in range(n_spec):
            with open(os.path.join(self._write_path, f"trajectory_species_{i}.dat"), 'w') as file:

                for j in range(n_traj):
                    header_str = f'# step = {steps[j]}, species = {i}, size = ' + dim_string + '\n'
                    file.write(header_str)
                    cur_frame = traj_log[j, i]

                    # Write the current 2D frame and a new line:
                    if self._integrator._dim == 1:
                        row_as_string = ' '.join(map(str, cur_frame))  # 1D just writes frame data
                        file.write(row_as_string + '\n')
                    elif self._integrator._dim == 2:
                        row_strings = [' '.join(map(str, row)) for row in cur_frame]
                        for row in row_strings:
                            file.write(row + '\n')

                # Then write the final_rho value:
                try:
                    header_str = f'# step = {steps[j + 1]}, species = {i}, size = ' + dim_string + '\n'
                    file.write(header_str)
                    cur_frame = final_rho[i]

                    # Write the current 2D frame and a new line:
                    if self._integrator._dim == 1:
                        row_as_string = ' '.join(map(str, cur_frame))  # 1D just writes frame data
                        file.write(row_as_string + '\n')
                    elif self._integrator._dim == 2:
                        row_strings = [' '.join(map(str, row)) for row in cur_frame]
                        for row in row_strings:
                            file.write(row + '\n')
                except IndexError:  # Need to clean up logging still
                    pass
                file.write('\n')

    @partial(jax.jit, static_argnums=(0,))
    def _log_energy(self, count, r):
        # Energy output format
        energy_values = jnp.array([
            count * self._system.dt,  # Time at end of step count+1
            self.average_free_energy(r),
            self.average_mass(r),
            count
        ], dtype=jnp.float32)
        return energy_values

    def run_system_no_logging(self, steps: int = None):
        """ This is for simple performance test-bedding. This can also be used to rapidly iterate an initial state to
        a specific point from which you may want to begin logging.
        """
        name = 'init_' + self._system.field_name
        rho_0 = getattr(self._system, name)  # Error will raise if not specified
        self._print_current_state("init_", 0, rho=rho_0)
        if steps is None:
            steps = self._steps

        def _evolve(_r, _):
            _r = self._system.evolve(_r)  # Evolve state first
            return _r, None

        rho_n, _ = jax.lax.scan(_evolve, rho_0, jnp.arange(steps))
        self._print_current_state("last_", 0, rho=rho_n)
        return rho_n

    @partial(jax.jit, static_argnums=(0, 2))
    def _evolve_n_steps(self, rho_0, steps):

        def _evolve(_r, _):
            _r = self._system.evolve(_r)  # Evolve state first
            return _r, None

        rho_n, _ = jax.lax.scan(_evolve, rho_0, jnp.arange(steps))
        return rho_n

    def debug_run_system(self, steps: int, log_dir: str):
        """ This is for simple performance test-bedding. This can also be used to rapidly iterate an initial state to
        a specific point from which you may want to begin logging.
        """
        import jax.profiler
        name = 'init_' + self._system.field_name
        rho_0 = getattr(self._system, name)  # Error will raise if not specified

        def _evolve(_r, _):
            _r = self._system.evolve(_r)  # Evolve state first
            return _r, None

        rho_warmup, _ = jax.lax.scan(_evolve, rho_0, jnp.arange(10))
        rho_warmup.block_until_ready()  # Wait for warm-up JIT & compute

        jax.profiler.start_trace(log_dir)
        rho_n, _ = jax.lax.scan(_evolve, rho_0, jnp.arange(steps))
        rho_n.block_until_ready()
        jax.profiler.stop_trace()
        return rho_n

class PINNManager:
    """ Read in TOML + Solve Cahn-Hilliard / Allen-Cahn via a PINN-based simulation. """

    def __init__(self, config, custom_energy=None, custom_initial_condition=None, custom_PINN=None):
        self._config = config
        self._write_path = config.get('write_path', r'./')  # Assumes same directory writing by default

        # Set RNG:
        self._rng_seed = int(config.get('steps', np.random.randint(1000000)))
        if 'steps' not in config:
            print(f'RNG seed not specified, using seed: {self._rng_seed}')

        # Setup the free energy model, integrator, and system based on if jax is being used:
        # The custom_energy and custom_initial_conditions might be None (which is fine) but are still passed in
        # The use of the TOML config dictates the use of these custom functions
        self._free_energy_model = _read_in_energy_model(config, config.get('free_energy'), custom_energy)
        # Note: We can simply "pass in" the PINN below, as we are really just grabbing the initial condition for
        # code reuse
        model_type = config.get('model', 'ch').lower()
        self.model_type = model_type
        self._system = _read_in_model(self._free_energy_model, config, model_type,
                                      None, self._rng_seed, custom_fn=custom_initial_condition)
        initial_order_params = self._system.get_initial_condition()  # (N_species, Nx, Ny, ...)
        # self.initial_condition = initial_order_params
        N_species = initial_order_params.shape[0]
        N = initial_order_params.shape[1]

        print('NOTE: Modifying the input initial condition...')
        densities = jnp.array([0.01])
        noise_amplitude = 0.02
        noise = jax.random.uniform(jax.random.PRNGKey(0), shape=(N_species, N, N)) - 0.5
        initial_field = densities[:, None, None] + noise_amplitude * noise
        self.initial_condition = initial_field

        self.N_species = N_species
        if custom_PINN is not None:
            self._network = custom_PINN
        else:
            self._network = self._read_in_network(self._system.dim, N_species, config)

        # Store init data from system:
        name = 'init_' + self._system.field_name
        self.init_field = getattr(self._system, name)

    @staticmethod
    def _read_in_network(dimensions, n_species, config):
        """ Reads in a specified PINN to replace the numerical integrator """
        network_type = config.get('network', 'mlp_fourier').lower()
        output_dimension = 2 * n_species  # The output must be 2 * n_species for (rho_1, ..., mu_1, ...)
        if network_type.lower() == 'mlp':
            layers = config.get('network_size', [128, 128, 128, 128])
            activation = config.get('activation_function', 'swish')
            activation_func = _read_in_activation_function(activation.lower())
            return MLP(dimensions+1, output_dimension, layers, activation_func, False)

        elif network_type.lower() == 'mlp_fourier':
            layers = config.get('network_size', [128, 128, 128, 128])
            activation = config.get('activation_function', 'tanh')
            activation_func = _read_in_activation_function(activation)
            fourier_dim = config.get('fourier_feature_dim', 32)
            fourier_scale = config.get('fourier_feature_scale', 1.0)
            trainable_b = config.get('trainable_B', False)
            return MLP(dimensions + 1, output_dimension, layers, activation_func, True, fourier_dim, fourier_scale,
                       trainable_b)

        else:
            raise Exception('Invalid network type specified, valid options are: `mlp`, `mlp_fourier`, ...')

    def solve(self, write_trajectory: bool = True):
        """ Trains the network to solve the Cahn-Hilliard or Allen-Cahn equation via a PINN approach. """
        start = time.time()
        assert len(self.initial_condition.shape) in [3], 'ERROR: PINN only setup for 2D problems (N_species,' \
                                                            ' Nx, Ny) as of now.'
        if self.model_type == 'ch':
            trained_params = train_ch(self._config, self._network, self._free_energy_model, self._system,
                                      self.N_species, self.initial_condition)
        else:
            trained_params = train_ac(self._config, self._network, self._free_energy_model, self._system,
                                      self.N_species, self.initial_condition)
        end = time.time()
        print(f'Training completed in {end - start:.2f} seconds.')

        # First export the final frame:
        self._export_final_frame_all_species(trained_params)

        # Export a video of each:
        if write_trajectory:
            print('NOTE: Writing trajectory file may take some time, please be patient...')
            num_frames = self._config.get('num_frames', 100)
            self._export_trajectories(trained_params, 1.0, num_frames)  # Animation -> t_final of 1.0 (normalied)

    def _export_final_frame_all_species(self, trained_params):
        """ Exports a final .dat frame of each N species """
        # Apply the trained params + output result of Cahn-Hilliard for final "time" along with a trajectory:
        t_final = 1.0
        x_bounds = [0.0, 1.0]
        y_bounds = [0.0, 1.0]
        grid_res = self.init_field.shape[1]  # Assuming square grid, Nx = Ny = grid_res
        x_space = jnp.linspace(x_bounds[0], x_bounds[1], grid_res)
        y_space = jnp.linspace(y_bounds[0], y_bounds[1], grid_res)
        xx, yy = jnp.meshgrid(x_space, y_space)
        tt = jnp.ones_like(xx) * t_final

        # Flatten the grid and stack to create model inputs (N, 3) for (x, y, t)
        xyt_eval = jnp.stack([xx.flatten(), yy.flatten(), tt.flatten()], axis=-1)

        # Apply the model with the trained parameters
        # The model will output predictions for [rho_1..N, mu_1..N]
        predictions = self._network.apply(trained_params, xyt_eval)

        # Extract and reshape the concentration (rho)
        rho_all_species = predictions[:, :self.N_species]

        for i in range(self.N_species):
            # Select the data for the current species 'i'
            rho_flat = rho_all_species[:, i]
            rho_final_grid = rho_flat.reshape((grid_res, grid_res))

            # Write out final grid:
            solution_grid_np = np.array(rho_final_grid)
            dim_string = f'{grid_res}x{grid_res}'
            header_str = f'# normalized_time = {t_final}, species = {i}, size = ' + dim_string + '\n'
            outpath = os.path.join(self._write_path, f"solution_species_{i}.dat")
            with open(outpath, 'w') as file:
                file.write(header_str)
                for row in solution_grid_np:
                    row_as_string = ' '.join(map(str, row))
                    file.write(row_as_string + '\n')

    def _export_trajectories(self, trained_params, t_final, num_frames):
        """ Exports trajectory files using the trained network parameters for each species """
        time_steps = np.linspace(0.0, t_final, num=num_frames)

        # Create basic objects:
        x_bounds = [0.0, 1.0]
        y_bounds = [0.0, 1.0]
        grid_res = self.init_field.shape[1]  # Assuming square grid, Nx = Ny = grid_res
        x_space = jnp.linspace(x_bounds[0], x_bounds[1], grid_res)
        y_space = jnp.linspace(y_bounds[0], y_bounds[1], grid_res)
        dim_string = f'{grid_res}x{grid_res}'
        all_species_to_export = {i: [] for i in range(self.N_species)}

        for i, t in enumerate(time_steps):
            xx, yy = jnp.meshgrid(x_space, y_space)
            tt = jnp.ones_like(xx) * t  # For each time-step
            xyt_eval = jnp.stack([xx.flatten(), yy.flatten(), tt.flatten()], axis=-1)

            predictions = self._network.apply(trained_params, xyt_eval)

            # Now extract rho for each species:
            for n in range(self.N_species):
                rho_flat = predictions[:, n]  # Assuming single species
                rho_grid = rho_flat.reshape((grid_res, grid_res))
                next_frame_string = f'# normalized_time = {t}, species = {n}, size = ' + dim_string + '\n'
                all_species_to_export[n].append((next_frame_string, rho_grid))

        # Now we will write out the trajectory files for each N-species from the dict:
        for species, frames in all_species_to_export.items():
            outpath = os.path.join(self._write_path, f"trajectory_species_{species}.dat")
            with open(outpath, 'w') as file:
                for f in frames:
                    header_str, rho_grid = f
                    file.write(header_str)

                    for row in rho_grid:
                        row_as_string = ' '.join(map(str, row))
                        file.write(row_as_string + '\n')


def _read_in_energy_model(config, free_energy, CustomEnergy):
    if free_energy.lower() == 'landau':
        return Landau(config)
    elif free_energy.lower() == 'simple_wertheim':
        return SimpleWertheim(config)
    elif free_energy.lower() == 'generic_wertheim':
        return GenericWertheim(config)
    elif free_energy.lower() == 'saleh' or free_energy.lower() == 'saleh_wertheim':
        return SalehWertheim(config)
    elif free_energy.lower() == 'custom':
        return CustomEnergy(config)

    else:
        raise Exception(f'Invalid free_energy specified: {free_energy}, valid options are: landau, custom')


def _read_in_model(model, config, model_name, integrator, rng_seed, custom_fn):
    if model_name.lower() == 'ch':
        return CahnHilliard(model, config, integrator, rng_seed, custom_fn=custom_fn)
    elif model_name.lower() == 'ac':
        return AllenCahn(model, config, integrator, rng_seed, custom_fn=custom_fn)
    else:
        raise Exception('Invalid model_name, valid options are: ch (Cahn-Hilliard), ac (Allen-Cahn)')

def _read_in_activation_function(activation_name: str):
    """ Returns the flax activation function based on a string name """
    if activation_name == 'tanh':
        return nn.tanh
    elif activation_name == 'relu':
        return nn.relu
    elif activation_name == 'swish':
        return nn.swish
    elif activation_name == 'gelu':
        return nn.gelu
    else:
        raise Exception('ERROR: Invalid activation function name')


if __name__ == '__main__':
    c = toml.load(r'../Examples/Landau/jax_long/input_magnetic_film.toml')
    SimulationManager(c)
