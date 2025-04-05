import numpy as np
import toml
from pfm.energy_models import Landau, jLandau
from pfm.integrators import ExplicitEuler, jExplicitEuler
from pfm.models import CahnHilliard, jCahnHilliard
import os

"""
Current Development TODO
========================
1) Get just a simple Euler and Landau model up and running
2) Implement numba into this CPU version
3) Then implement other energy models, as we can worry about integration schemes later
4) Inform Bex / Lainie of this project, then work on JAX version
5) Write arXiv paper w/ results and performance comparison with C++ code
"""

class SimulationManager:
    def __init__(self, config):
        self._steps = config.get('steps', 100)
        self._print_mass_every = config.get('print_every', 10)
        self._print_trajectory_strategy = config.get('print_trajectory_strategy', 'linear').lower()
        if self._print_trajectory_strategy == 'linear':
            self._print_trajectory_every = config.get("print_trajectory_every", 100)
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
        self._USE_JAX = config.get('use_jax', False)
        if not self._USE_JAX:
            print('Note: Without jax, these simulations will take quite a long time. You will only be able to run '
                  'simple simulations for low amounts of time steps.')
        self._free_energy_model = self._read_in_energy_model(config, config.get('free_energy'), self._rng_seed,
                                                             self._USE_JAX)
        self._integrator = self._read_in_integrator(self._free_energy_model, config, config.get('integrator'),
                                                    self._rng_seed, self._USE_JAX)
        self._system = self._read_in_model(self._free_energy_model, config, config.get('model', 'ch'),
                                           self._integrator, self._rng_seed, self._USE_JAX)

        # Setup simulation trajectory tracking:
        num_species = self._free_energy_model.N_species()
        self._trajectories = []
        if self._print_trajectory_every > 0 or hasattr(self, '_log_n0'):  # Check if log printing is enabled
            for i in range(num_species):
                def_name = os.path.join(self._write_path, f"trajectory_{i}.dat")
                self._trajectories.append(open(def_name, "w"))
        self._traj_printed = 0

    def __del__(self):
        # Override default garbage collection so traj files are closed
        self.close()

    def close(self):
        for traj in self._trajectories:
            if traj:
                traj.close()

    @staticmethod
    def _read_in_energy_model(config, free_energy, rng_seed, USE_JAX):
        if free_energy.lower() == 'landau' and not USE_JAX:
            rng = np.random.default_rng(rng_seed)
            return Landau(config, rng)

        elif free_energy.lower() == 'landau' and USE_JAX:
            return jLandau(config)

        else:
            raise Exception('Invalid free_energy specified in the config, valid options are: landau, ')

    @staticmethod
    def _read_in_integrator(model, config, integrator_name, rng_seed, USE_JAX):
        if integrator_name.lower() == 'euler' and not USE_JAX:
            rng = np.random.default_rng(rng_seed)
            return ExplicitEuler(model, config, rng)

        elif integrator_name.lower() == 'euler' and USE_JAX:
            return jExplicitEuler(model, config)
        else:
            raise Exception('Invalid integrator scheme, valid options are: euler, ')

    @staticmethod
    def _read_in_model(model, config, model_name, integrator, rng_seed, USE_JAX):
        if model_name.lower() == 'ch' and not USE_JAX:
            rng = np.random.default_rng(rng_seed)
            return CahnHilliard(model, config, integrator, rng)
        elif model_name.lower() == 'ch' and USE_JAX:
            return jCahnHilliard(model, config, integrator, rng_seed)
        else:
            raise Exception('Invalid model_name, valid options are: ch (Cahn-Hilliard), ')

    def _print_current_state(self, prefix, t, rho=None):
        print(f"{prefix} state at time {t}")
        num_species = self._free_energy_model.N_species()
        for i in range(num_species):
            filename = f"{prefix}{i}.dat"
            fp = os.path.join(self._write_path, filename)
            with open(fp, "w") as output:
                if self._USE_JAX:
                    self._system.print_species_density(i, output, t, rho)
                else:
                    self._system.print_species_density(i, output, t)
        # Maybe add a method to CahnHilliard to print total density if needed?
        # Something like below?
        # with open(f"{prefix}density.dat", "w") as output:
        #     self._system.print_total_density(output, t)

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
        if self._USE_JAX:
            return self._system.average_free_energy(rho)
        else:
            return self._system.average_free_energy()

    def average_mass(self, rho=None):
        """ Cahn-Hilliard (or Allen-Cahn when implemented) will simply calculate this based on stored rho values """
        if self._USE_JAX:
            return self._system.average_mass(rho)
        else:
            return self._system.average_mass()

    """
    Main run methods below
    """
    def _run_jax(self):
        """ Runs the jax-version of the solver. Here, we need to treat rho as a variable instead of at the state level
        to properly use jax.
        """
        rho_0 = self._system.init_rho
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

    def _run_numpy(self):
        """ Runs the standard (slow) numpy code which can be used for simple implementation verification """
        self._print_current_state("init_", 0)
        fp = os.path.join(self._write_path, 'energy.dat')
        with open(fp, "w") as mass_output:
            for t in range(self._steps):
                if self._should_print_last(t):
                    self._print_current_state("last_", t)

                if self._should_print_traj(t):
                    num_species = self._free_energy_model.N_species()
                    for i in range(num_species):
                        self._system.print_species_density(i, self._trajectories[i], t)
                    self._traj_printed += 1

                if self._print_mass_every > 0 and t % self._print_mass_every == 0:
                    output_line = (f"{t * self._system.dt:.5f} {self.average_free_energy():.8f} "
                                   f"{self.average_mass():.5f} {t}")
                    mass_output.write(output_line + "\n")
                    print(output_line)

                self._system.evolve()

        self._print_current_state("last_", self._steps)

    def run(self):
        """ Main function of the Simulation Manager, runs the simulation """
        if self._USE_JAX:
            self._run_jax()
        else:
            self._run_numpy()



if __name__ == '__main__':
    c = toml.load(r'../Examples/Landau/input_landau.toml')
    SimulationManager(c)
