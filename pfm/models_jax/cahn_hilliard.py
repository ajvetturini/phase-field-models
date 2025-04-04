"""
Cahn-Hilliard implementation
"""
import numpy as np

class CahnHilliard:
    def __init__(self, free_energy_model, config, integrator, rng):
        self._rng = rng
        self.N = config.get('N')
        self.k_laplacian = config.get('k', 1.0)
        self.dt = config.get('dt')
        self.M = config.get('M', 1.0)
        self.dx = config.get('dx', 1.0)
        self._internal_to_user = config.get('distance_scaling_factor', 1.0)
        self._user_to_internal = 1.0 / self._internal_to_user
        self.dim = config.get('dim', 2)
        if self.dim <= 0 or self.dim > 2:
            raise Exception('Unable to proceed, currently only support for 1D and 2D is implemented')

        log2N = np.log2(self.N)
        if np.ceil(log2N) != np.floor(log2N):
            raise ValueError("N should be a power of 2")

        self.N_minus_one = self.N - 1
        self.bits = int(log2N)

        self.grid_size = self.N ** self.dim
        self._grid_size_str = "x".join([str(self.N)] * self.dim)

        num_species = free_energy_model.N_species()
        shape = tuple([num_species] + [self.N] * self.dim)
        rho = np.zeros(shape)

        if "load_from" in config:
            filename = config.get('load_from')
            with open(filename, 'r') as load_from:
                for s in range(num_species):
                    if self.dim == 1:
                        for idx in range(self.N):
                            rho[s, idx] = float(load_from.readline().strip())
                    elif self.dim == 2:
                        coords = [0, 0]
                        for coords[1] in range(self.N):
                            for coords[0] in range(self.N):
                                idx = self.cell_idx(coords)
                                rho[s, coords[1], coords[0]] = float(load_from.readline().strip())
                    else:
                        raise ValueError(f"Unsupported number of dimensions {self.dim}")
        elif "initial_density" in config:
            densities = config.get('initial_density')
            initial_A = config.get('initial_A', 1e-2)
            initial_N_peaks = config.get('initial_N_peaks', 0)
            initial_k = 2 * np.pi * initial_N_peaks / self.N  # Wave vector of modulation
            for bin in range(self.grid_size):
                modulation = initial_A * np.cos(initial_k * bin)
                for i in range(num_species):
                    random_factor = (self._rng.random() - 0.5 if initial_N_peaks == 0 else
                                     1.0 + 0.02 * (self._rng.random() - 0.5))
                    average_rho = densities[i]
                    coords = self.fill_coords(bin)
                    if self.dim == 1:
                        rho[i, bin] = average_rho * (1.0 + 2.0 * modulation * random_factor)
                    elif self.dim == 2:
                        rho[i, coords[1], coords[0]] = average_rho * (1.0 + 2.0 * modulation * random_factor)
        else:
            raise ValueError("Either 'initial_density' or 'load_from' should be specified")

        self.dx *= self._user_to_internal
        self.M /= self._user_to_internal
        self.k_laplacian *= self._user_to_internal ** 5
        rho /= self._user_to_internal ** 3

        self.V_bin = self.dx ** 3

        self.integrator = integrator
        self.integrator.set_initial_rho(rho)

        self._output_ready = False
        self._d_vec_size = 0
        self._grid_size_str = ""


    def fill_coords(self, idx):
        coords = []
        for d in range(self.dim):
            coords.append(idx & self.N_minus_one)
            idx >>= self.bits
        return coords

    def cell_idx(self, coords):
        idx = 0
        multiply_by = 1
        for d in range(self.dim):
            idx += coords[d] * multiply_by
            multiply_by <<= self.bits
        return idx

    def gradient(self, field, species, coords):
        grad = np.zeros(self.dim)
        for d in range(self.dim):
            coords_plus = coords[:]
            coords_minus = coords[:]
            coords_plus[d] = (coords[d] + 1) & self.N_minus_one
            coords_minus[d] = (coords[d] - 1 + self.grid_size) & self.N_minus_one
            if self.dim == 1:
                grad[d] = (field[species, coords_plus[0]] - field[species, coords_minus[0]]) / (2 * self.dx)
            else:
                grad[d] = (field[species, coords_plus[d % 2], coords_plus[(d+1)%2]] - field[species, coords_minus[d % 2], coords_minus[(d+1)%2]]) / (2 * self.dx)
        return grad

    def evolve(self):
        self.integrator.evolve()

    def average_mass(self):
        rho = self.integrator.rho()
        mass = 0.0
        for i in range(np.prod(rho.shape[1:])):
            mass += np.sum(rho[:, self.fill_coords(i)[1], self.fill_coords(i)[0]])
            if np.isnan(mass):
                raise ValueError(f"Encountered a nan while computing the total mass (bin {i})")
        return mass * self.V_bin / np.prod(rho.shape[1:])

    def average_free_energy(self):
        rho = self.integrator.rho()
        fe = 0.0
        for i in range(np.prod(rho.shape[1:])):
            coords = self.fill_coords(i)
            interfacial_contrib = 0.0
            for species in range(rho.shape[0]):
                rho_grad = self.gradient(rho, species, coords)
                interfacial_contrib += self.k_laplacian * np.sum(rho_grad ** 2)
            fe += self.model.bulk_free_energy(rho[:, coords[1], coords[0]]) + interfacial_contrib
        return fe * self.V_bin / np.prod(rho.shape[1:])

    def print_species_density(self, species, output, t):
        output.write(f"# step = {t}, t = {t * self.dt:.5f}, size = {self._grid_size_str}\n")
        rho = self.integrator.rho()
        for idx in range(np.prod(rho.shape[1:])):
            coords = self.fill_coords(idx)
            if idx > 0:
                modulo = self.N
                for d in range(1, self.dim):
                    if idx % modulo == 0:
                        output.write("\n")
                    modulo <<= self.bits
            if self.dim == 1:
                output.write(f"{self._density_to_user(rho[species, coords[0]])} ")
            else:
                output.write(f"{self._density_to_user(rho[species, coords[1], coords[0]])} ")
        output.write("\n")

    def _density_to_user(self, v):
        return v / (self._internal_to_user ** 3)
