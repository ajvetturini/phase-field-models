"""
The base FreeEnergyModel class that will pull in the relevent energy models (e.g., landau) to construct a total
FreeEnergyModel used in Cahn Hilliard simulations.
"""

class FreeEnergyModel:

    def __init__(self, config):
        self._config = config
        self._inverse_scaling_factor = 1.0 / config.get("distance_scaling_factor", 1.0)
        self._density_conversion_factor = self._inverse_scaling_factor ** 3

    def average_energy(self, u):
        raise NotImplementedError('Not implemented here.')

    def N_species(self):
        raise NotImplementedError('N_species must be implemented in derived classes.')

    def der_bulk_free_energy_expansive(self, species, rho_species):
        raise NotImplementedError('Not implemented here.')

    def der_bulk_free_energy_contractive(self, species, rho_species):
        raise NotImplementedError('Not implemented here.')

    def der_bulk_free_energy(self, species, rho_species):
        # Derivative of bulk free energy
        raise NotImplementedError('Not implemented here.')

    def der_bulk_free_energy_autodiff(self, species, rho_species):
        raise NotImplementedError('Not implemented here.')

    def bulk_free_energy(self, rho_species):
        # Bulk free energy
        raise NotImplementedError('Not implemented here.')

