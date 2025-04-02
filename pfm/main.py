import numpy as np
import toml
from pfm.energy_models.landau import Landau  # Eventually update init to have all the necessary imports...
from pfm.models.cahn_hilliard import CahnHilliard

class SimulationManager:
    def __init__(self, config):
        self._steps = config.get('steps', 100)
        self._print_mass_every = config.get('print_every', 10)
        self._print_trajectory_every = config.get("print_trajectory_every", 100)
        self._config = config
        self._free_energy_model = self._read_in_energy_model(config, config.get('free_energy'))
        self._system = CahnHilliard(config, self._free_energy_model)
        self._trajectories = [open(f"traj_{i}.dat", "w") for i in range(1)]  # Only one species here.
        self._traj_printed = 0

    def __del__(self):
        self.close()

    def close(self):
        for traj in self._trajectories:
            if traj:
                traj.close()

    @staticmethod
    def _read_in_energy_model(config, model_name):
        if model_name.lower() == 'landau':
            return Landau(config)
        else:
            raise Exception('Invalid model_name specified in the config.')

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
