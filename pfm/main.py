import numpy as np
import toml
import jax
import jax.numpy as jnp
from pfm.energy_models import Landau, SimpleWertheim, GenericWertheim, SalehWertheim
from pfm.integrators import ExplicitEuler
from pfm.models import CahnHilliard, AllenCahn
import os
from functools import partial
import time
import sys


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
        self._free_energy_model = self._read_in_energy_model(config, config.get('free_energy'), custom_energy)
        self._integrator = self._read_in_integrator(self._free_energy_model, config, config.get('integrator', 'euler'))
        self._system = self._read_in_model(self._free_energy_model, config, config.get('model', 'ch'),
                                           self._integrator, self._rng_seed, custom_fn=custom_initial_condition)
        self._traj_printed = 0
        self._trajectories = []
        self._custom_initial_condition = custom_initial_condition

    def close(self):
        for traj in self._trajectories:
            if traj:
                traj.close()

    @staticmethod
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

    @staticmethod
    def _read_in_integrator(model, config, integrator_name):
        if integrator_name.lower() == 'euler':
            return ExplicitEuler(model, config)
        else:
            raise Exception('Invalid integrator scheme, valid options are: euler, ')

    @staticmethod
    def _read_in_model(model, config, model_name, integrator, rng_seed, custom_fn):
        if model_name.lower() == 'ch':
            return CahnHilliard(model, config, integrator, rng_seed, custom_fn=custom_fn)
        elif model_name.lower() == 'ac':
            return AllenCahn(model, config, integrator, rng_seed, custom_fn=custom_fn)
        else:
            raise Exception('Invalid model_name, valid options are: ch (Cahn-Hilliard), ac (Allen-Cahn)')

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
        """ Cahn-Hilliard (or Allen-Cahn when implemented) will simply calculate this based on stored rho values """
        return self._system.average_free_energy(rho)

    def average_mass(self, rho=None):
        """ Cahn-Hilliard (or Allen-Cahn when implemented) will simply calculate this based on stored rho values """
        return self._system.average_mass(rho)

    """
    Main run method
    """

    def _run_cpu(self):
        """ Main function of the Simulation Manager, runs the simulation. This is a "slow run" which does not leverage
                 jax outside of jit-ting the process. This should result in similar to CPU based C++ implementation, but is
                 not well-suited for very large (1e9-1e10) timestep simulations
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

    def _run_jax(self):
        """ uses evolve_n_steps and proper logging in arrays to store information """
        name = 'init_' + self._system.field_name
        rho_0 = getattr(self._system, name)  # Error will raise if not specified

        assert self._print_trajectory_every > self._print_mass_every, 'ERROR: Print trajectory should be > print_energy'

        # Determine the number of logging steps for trajectory and energy
        num_traj_log_points = int(max(1, self._steps // self._print_trajectory_every))
        num_energy_log_points = int(max(1, self._steps // self._print_mass_every))

        max_log_points = 250
        if num_traj_log_points > max_log_points:
            print(f'print_trajectory_every set too high for JAX. Limiting trajectory logs to {max_log_points} states.')
            num_traj_log_points = max_log_points
            self._print_trajectory_every = max(1, self._steps // num_traj_log_points)

        if num_energy_log_points > max_log_points:
            print(f'print_mass_every set too high for JAX. Limiting energy logs to {max_log_points} evaluations.')
            num_energy_log_points = max_log_points
            self._print_mass_every = max(1, self._steps // num_energy_log_points)

        # init rho and logging arrays
        self._print_current_state("init_", 0, rho=rho_0)  # Print init state
        rho_n = rho_0  # init
        steps_per_traj_log = self._print_trajectory_every
        energy_logs_per_traj = steps_per_traj_log // self._print_mass_every
        if energy_logs_per_traj == 0 and self._print_trajectory_every >= self._print_mass_every:
            energy_logs_per_traj = 1
        elif self._print_mass_every > self._print_trajectory_every:
            raise ValueError(
                "print_mass_every should be less than or equal to print_trajectory_every for proper logging.")

        N, Ns, dim = self._system.N, self._free_energy_model.N_species(), self._system.dim
        if dim == 1:
            dim_str = f'{N}'
        elif dim == 2:
            dim_str = f'{N}x{N}'
        elif dim == 3:
            dim_str = f'{N}x{N}x{N}'
        else:
            raise Exception('Invalid value of dimension')

        energy_log = [self._log_energy(0, rho_0)]
        traj_log = [np.array(rho_0, dtype=np.float32)]
        start = time.time()
        time_points = [start]  # Keep track of runtime

        current_step = 0
        for i in range(num_traj_log_points):
            # Number of steps to evolve until the next trajectory log
            n_steps = int(min(steps_per_traj_log, self._steps - current_step))

            # Pre-allocate space for energy logs within this trajectory segment
            num_energy_logs = int(
                min(
                    energy_logs_per_traj,
                    (self._steps - current_step) // self._print_mass_every if self._print_mass_every > 0 else 1
                )
            )
            energy_at_segment = jnp.zeros((num_energy_logs, 4), dtype=jnp.float32) if (
                    num_energy_logs > 0) else jnp.zeros((0, 4), dtype=jnp.float32)

            # Evolve the system and collect energy data
            rho_n, e_n = self._evolve_n_steps(rho_n, n_steps, energy_at_segment, self._print_mass_every)
            if num_energy_logs > 0:
                energy_log.extend(np.array(e_n, dtype=np.float32))

            # Store the trajectory point
            traj_log.append(np.array(rho_n, dtype=np.float32))
            current_step += n_steps
            time_points.append(current_step)

            if current_step >= self._steps:
                break

        # Print final state and logs
        energy_log = np.array(energy_log)
        traj_log = np.array(traj_log)
        self._print_current_state("last_", self._steps, rho=rho_n)

        print('Beginning write out of final files, this may take a moment...')
        self._write_output_jax_arrays(traj_log, energy_log, np.array(time_points), dim_str, rho_n)

    def run(self, override_use_jax: bool = False):
        """ Top level method to run a simulation """
        devices = jax.devices()
        accelerator_found = False
        for d in devices:
            if d.device_kind.lower() == 'gpu' or d.device_kind.lower() == 'tpu':
                accelerator_found = True
                break
            elif d.device_kind.lower() not in ['gpu', 'tpu', 'cpu']:
                raise Exception(f'Unknown hardware device: {d.device_kind}')

        if accelerator_found:
            self._run_jax()
        else:
            self._run_cpu()  # Still leverages JIT / JAX, but write out happens a bit differently

    def _write_output_jax_arrays(self, traj_log, energy_log, steps, dim_string, final_rho):
        """ Writes a simple ASCII file of the trajectory and energies tracked during the simulation """
        shaped_energy = energy_log.reshape((-1, energy_log.shape[2]))
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

    @partial(jax.jit, static_argnums=(0, 2))
    def _evolve_n_steps(self, rho, nsteps, energy_at_step, energy_every):
        """ Steps the integrator some N number of timesteps using a jax.lax.scan """

        def _evolve(carry, count):
            r, el, ec = carry
            r = self._system.evolve(r)  # Evolve state first
            log_energy_cond = (jnp.mod(count, energy_every) == 0)

            (el, ec) = jax.lax.cond(
                log_energy_cond,
                lambda: (el.at[ec].set(self._log_energy(count, r)), ec + 1),  # Branch that logs
                lambda: (el, ec)  # Branch that does nothing
            )

            return (r, el, ec), None

        (r_n, el, ec), _ = jax.lax.scan(_evolve, (rho, energy_at_step, 0), jnp.arange(nsteps), nsteps)
        return r_n, el

    def run_system_no_logging(self, steps: int = None):
        """ This is for simple performance test-bedding. This can also be used to rapidly iterate an initial state to
        a specific point from which you may want to begin logging.
        """
        name = 'init_' + self._system.field_name
        rho_0 = getattr(self._system, name)  # Error will raise if not specified
        if steps is None:
            steps = self._steps

        rho_n = self._evolve_n_steps(rho_0, steps)
        return rho_n


if __name__ == '__main__':
    c = toml.load(r'../Examples/Landau/jax_long/input_magnetic_film.toml')
    SimulationManager(c)
