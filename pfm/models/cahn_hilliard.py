"""
Cahn-Hilliard implementation
"""
import numpy as np

class CahnHilliard:
    def __init__(self, free_energy_model, config, rng):
        self._rng = rng
        self.dt = config.get('dt', 0.001)
        self.N = config.get('N', 64)
        self.initial_density = config.get('initial_density', 0.01)
        self.dim = config.get('dim', 2)
        if self.dim <= 0 or self.dim > 2:
            raise Exception('Unable to proceed, currently only support for 1D and 2D is implemented')
        shape = tuple([self.N] * self.dim)
        self.grid = self._rng.uniform(0, 1, shape) * self.initial_density
        self.grid_size = np.pow(self.N, self.dim)
        self.model = free_energy_model
        self.N_minus_one = self.N - 1
        self.k_laplacian = config.get('k_laplacian', 1.0)
        self.M = config.get('M', 1.0)
        self.dx = config.get('dx', 1.0)
        self.V_bin = config.get('V_bin', 0.0)  # Default value, adjust as needed

    def evolve(self):
        # Implement the Cahn-Hilliard evolution logic here
        lap_u = self.laplacian(self.grid)
        dfdu = self.model.average_free_energy(self.grid)  # Free energy derivative
        lap_dfdu = self.laplacian(dfdu)
        self.grid += -self.dt * self.M * self.laplacian(lap_u - self.k_laplacian * lap_dfdu)

    def average_mass(self):
        return np.mean(self.grid)

    def average_free_energy(self):
        return np.mean(self.model.average_free_energy(self.grid))

    def laplacian(self, u):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (self.dx * self.dx) + \
                          (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (self.dx * self.dx)
        return lap

    def print_species_density(self, species, output, t):
        output.write(f"Time: {t}, Species {species}, Density: {np.mean(self.grid)}\n")  # example
