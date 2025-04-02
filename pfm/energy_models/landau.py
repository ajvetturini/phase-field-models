from pfm.energy_models.free_energy_model import FreeEnergyModel

class Landau(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)  # First init the relevent FreeEnergyModel config options
        self.epsilon = config.get('epsilon', 1.0)

    def average_free_energy(self, grid):
        """ Takes the density grid and evaluates the average free energy """



