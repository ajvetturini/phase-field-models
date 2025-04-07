import numpy as np
import toml
import jax
import jax.numpy as jnp
from pfm.energy_models import Landau, MagneticFilm
from pfm.integrators import ExplicitEuler
from pfm.models import CahnHilliard, AllenCahn
import os
from functools import partial


class SimulationManager:
    def __init__(self, config, custom_initial_condition=None):
        self._steps = config.get('steps', 100)
        self._print_mass_every = config.get('print_every', 10)
        self._print_trajectory_strategy = config.get('print_trajectory_strategy', 'linear').lower()
        if self._print_trajectory_strategy == 'linear':
            self._print_trajectory_every = config.get("print_trajectory_every")  # Must be specified to prevent
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
        self._rng_seed = config.get('steps', np.random.randint(1000000))
        if 'steps' not in config:
            print(f'RNG seed not specified, using seed: {self._rng_seed}')

        # Setup the free energy model, integrator, and system based on if jax is being used:
        self._free_energy_model = self._read_in_energy_model(config, config.get('free_energy'))
        self._integrator = self._read_in_integrator(self._free_energy_model, config, config.get('integrator'))
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
    def _read_in_energy_model(config, free_energy):
        if free_energy.lower() == 'landau':
            return Landau(config)
        elif free_energy.lower() == 'magnetic_film':
            return MagneticFilm(config)

        else:
            raise Exception('Invalid free_energy specified in the config, valid options are: landau, magnetic_film')

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
    def run(self):
        """ Main function of the Simulation Manager, runs the simulation. This is a "slow run" which does not leverage
         jax outside of jit-ting the process. This should result in similar to CPU based C++ implementation, but is
         not well-suited for very large (1e9-1e10) timestep simulations.
         """
        # Setup simulation trajectory tracking:
        num_species = self._free_energy_model.N_species()
        if self._print_trajectory_every > 0 or hasattr(self, '_log_n0'):  # Check if log printing is enabled
            for i in range(num_species):
                def_name = os.path.join(self._write_path, f"trajectory_{i}.dat")
                self._trajectories.append(open(def_name, "w"))

        try:
            rho_0 = self._system.init_rho
        except AttributeError:
            rho_0 = self._system.init_phi

        if self._config.get('verbose', True):
            self._print_current_state("init_", 0, rho=rho_0)
        fp = os.path.join(self._write_path, 'energy.dat')
        rho_n = rho_0  # init
        with open(fp, "w") as mass_output:
            for t in range(self._steps):
                if self._should_print_last(t) and self._config.get('verbose', True):
                    self._print_current_state("last_", t, rho=rho_n)

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
            self._print_current_state("last_", self._steps, rho=rho_n)

    def run_jax(self):
        """ uses evolve_n_steps and proper logging in arrays to store information """
        try:
            rho_0 = self._system.init_rho
        except AttributeError:
            rho_0 = self._system.init_phi

        # We will have to pre-allocate space in this version to store the trajectory and mass outputs:
        num_mass_states = max(1, self._steps // self._print_mass_every)
        num_traj_states = max(1, self._steps // self._print_trajectory_every)
        if num_mass_states > 250:
            print('Print_mass_every set too high for JAX. For memory purposes, we limit this to 1000 energy evals.')
            num_mass_states = 250
            self._print_mass_every = max(self._steps, self._steps // num_mass_states)

        if num_traj_states > 250:
            print('print_trajectory_every set too high for JAX. For memory purposes, we limit this to 250 states.')
            num_traj_states = 250
            self._print_trajectory_every = max(self._steps, self._steps // num_traj_states)

        # init rho and logging arrays
        self._print_current_state("init_", 0, rho=rho_0)  # Print init state
        rho_n = rho_0  # init
        num_energy_logs_per_traj_store = self._print_trajectory_every // self._print_mass_every
        leading_dim = max(1, num_mass_states // num_energy_logs_per_traj_store)
        energy_log = np.zeros((leading_dim, num_energy_logs_per_traj_store, 4), dtype=np.float32)

        N, Ns, dim = self._system.N, self._free_energy_model.N_species(), self._system.dim
        if dim == 1:
            dim_str = f'{N}'
            traj_log = np.zeros((num_traj_states, Ns, N,), dtype=np.float16)
        elif dim == 2:
            dim_str = f'{N}x{N}'
            traj_log = np.zeros((num_traj_states, Ns, N, N), dtype=np.float16)
        elif dim == 3:
            dim_str = f'{N}x{N}x{N}'
            traj_log = np.zeros((num_traj_states, Ns, N, N, N), dtype=np.float16)
        else:
            raise Exception('Invalid value of dimension')

        ns = max(1, self._steps // num_traj_states)
        for i in range(num_traj_states):
            # Blank energy array to store output
            energy_at_step = jnp.zeros((num_energy_logs_per_traj_store, 4), dtype=jnp.float32)
            rho_n, e_n = self._evolve_n_steps(rho_n, ns, energy_at_step, self._print_mass_every)
            energy_log[i] = e_n
            traj_log[i] = np.array(rho_n, dtype=np.float16)

        # Print final state and logs
        self._print_current_state("last_", self._steps, rho=rho_n)

        print('Beginning write out of final files, this may take a moment...')
        self._write_output_jax_arrays(traj_log, energy_log, np.arange(num_traj_states+1)*ns, dim_str, rho_n)


    def _write_output_jax_arrays(self, trajectory_lost, energy_log, steps, dim_string, final_rho):
        """ Writes a simple ASCII file of the trajectory and energies tracked during the simulation """
        shaped_energy = energy_log.reshape((-1, energy_log.shape[2]))
        write_list = []
        s1, s2 = 0, 0
        for i in shaped_energy:
            write_list.append(f'{s1:.5f} {i[1]:.8f} {i[2]:.5f} {s2:d}')
            if s1 == 0:
                add1 = shaped_energy[1][0]
                add2 = int(shaped_energy[1][-1])
            s1 += add1
            s2 += add2

        with open(os.path.join(self._write_path, 'energy.dat'), 'w') as file:
            for item in write_list:
                file.write(item + '\n')

        # Repeat for trajectory:
        n_traj, n_spec = trajectory_lost.shape[0], trajectory_lost.shape[1]
        for i in range(n_spec):
            with open(os.path.join(self._write_path, f"trajectory_species_{i}.dat"), 'w') as file:

                for j in range(n_traj):
                    header_str = f'# step = {steps[j]}, species = {i}, size = ' + dim_string + '\n'
                    file.write(header_str)
                    cur_frame = trajectory_lost[j, i]

                    # Write the current 2D frame and a new line:
                    row_strings = [' '.join(map(str, row)) for row in cur_frame]
                    for row in row_strings:
                        file.write(row + '\n')

                # Then write the final_rho value:
                header_str = f'step = {steps[j+1]}, species = {i}, size = ' + dim_string
                file.write(header_str)
                cur_frame = final_rho[i]

                # Write the current 2D frame and a new line:
                row_strings = [' '.join(map(str, row)) for row in cur_frame]
                for row in row_strings:
                    file.write(row + '\n')
                file.write('\n')



    @partial(jax.jit, static_argnums=(0, 2))
    def _evolve_n_steps(self, rho, nsteps, energy_at_step, energy_every):
        """ Steps the integrator some N number of timesteps using a jax.lax.scan """

        def _evolve(carry, count):
            r, el, ec = carry
            r = self._system.evolve(r)  # Evolve state first
            log_energy_cond = (jnp.mod(count, energy_every) == 0)

            def _log_energy():
                # Energy output format
                energy_values = jnp.array([
                    count * self._system.dt,  # Time at end of step count+1
                    self.average_free_energy(r),
                    self.average_mass(r),
                    count
                ], dtype=jnp.float32)
                return energy_values

            (el, ec) = jax.lax.cond(
                log_energy_cond,
                lambda: (el.at[ec].set(_log_energy()), ec+1),  # Branch that logs
                lambda: (el, ec)  # Branch that does nothing
            )

            return (r, el, ec), None

        (r_n, el, ec), _ = jax.lax.scan(_evolve, (rho, energy_at_step, 0), jnp.arange(nsteps), nsteps)
        return r_n, el

    def run_system_no_logging(self):
        try:
            rho_0 = self._system.init_rho
        except AttributeError:
            rho_0 = self._system.init_phi

        rho_n = self._evolve_n_steps(rho_0, self._steps)
        return rho_n



if __name__ == '__main__':
    c = toml.load(r'../Examples/Landau/jax_long/input_magnetic_film.toml')
    SimulationManager(c)
