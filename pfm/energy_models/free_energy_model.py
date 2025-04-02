"""
The base FreeEnergyModel class that will pull in the relevent energy models (e.g., landau) to construct a total
FreeEnergyModel used in Cahn Hilliard simulations.
"""

class FreeEnergyModel:

    def __init__(self, config):
        self._config = config
