import numpy as np
import toml
from pfm.energy_models import Landau
from pfm.integrators import Euler
from pfm.models.cahn_hilliard import CahnHilliard

"""
Current Development TODO
========================
1) I implemented base_integrator and eulerCPU. I should implement the FreeEnergyModel and Laudau code
2) Make sure I can handle multi-species, make sure that is handled properly (need to find where this is at in the code)
3) Get just a simple Euler and Landau model up and running
4) Then implement other energy models, as we can worry about integration schemes later
"""

class SimulationManager:
    def __init__(self, config):
        self._steps = config.get('steps', 100)
        self._print_mass_every = config.get('print_every', 10)
        self._print_trajectory_every = config.get("print_trajectory_every", 100)
        self._config = config
        # Set RNG:
        self._rng_seed = config.get('steps', np.random.randint(1000000))
        self._rng = np.random.default_rng(self._rng_seed)  # Using Generator, pass this into classes to use RNG

        self._free_energy_model = self._read_in_energy_model(config, config.get('free_energy'), self._rng)
        self._integrator = self._read_in_integrator(self._free_energy_model, config, config.get('integrator'), self._rng)
        self._system = self._read_in_model(self._free_energy_model, config, config.get('model', 'ch'), self._rng)
        self._trajectories = [open(f"traj_{i}.dat", "w") for i in range(1)]  # What is this?
        self._traj_printed = 0



    def __del__(self):
        self.close()

    def close(self):
        for traj in self._trajectories:
            if traj:
                traj.close()

    @staticmethod
    def _read_in_energy_model(config, free_energy, rng):
        if free_energy.lower() == 'landau':
            return Landau(config, rng)
        else:
            raise Exception('Invalid free_energy specified in the config, valid options are: landau, ')

    @staticmethod
    def _read_in_integrator(model, config, integrator_name, rng):
        if integrator_name.lower() == 'euler':
            return Euler(model, config, rng)
        else:
            raise Exception('Invalid integrator scheme, valid options are: euler, ')

    @staticmethod
    def _read_in_model(model, config, model_name, rng):
        if model_name.lower() == 'ch':
            return CahnHilliard(model, config, rng)
        else:
            raise Exception('Invalid model_name, valid options are: ch (Cahn-Hilliard), ')

    def _print_current_state(self, prefix, t):
        print(f"{prefix} state at time {t}")
        # Add code to print current simulation state

    @property
    def average_free_energy(self):
        return np.mean(self._free_energy_model.average_free_energy(self._system.grid))

    @property
    def average_mass(self):
        return np.mean(self._system.grid)

    def run(self):
        """ Main function of the Simulation Manager """
        self._print_current_state("init_", 0)

        with open("energy.dat", "w") as mass_output:
            for t in range(self._steps):
                if self._print_mass_every > 0 and t % self._print_mass_every == 0:
                    output_line = f"{t * self._system.dt:.5f} {self.average_free_energy():.8f} {self.average_mass():.5f} {t}"
                    mass_output.write(output_line + "\n")
                    print(output_line)
                self._system.evolve()

        self._print_current_state("last_", self._steps)


if __name__ == '__main__':
    c = toml.load(r'../Examples/Landau/input_landau.toml')
    SimulationManager(c)
