import numpy as np
from pfm.integrators.base_integrator import Integrator

class Euler(Integrator):

    def __init__(self, model, config, rng):
        super().__init__(model, config)
        self._N_per_dim_minus_one = self._N_per_dim - 1
        self._log2_N_per_dim = int(np.log2(self._N_per_dim))
        self._rng = rng

    def evolve(self):
        # Create rho_der array to store time derivatives
        rho_der = np.zeros_like(self._rho)

        # Calculate time derivatives
        for idx in range(self._N_bins):
            coords = self._fill_coords(idx)
            rho_species = self._rho[tuple([slice(None)] + coords)] #get species at index
            for species in range(self._model.N_species):
                rho_der[tuple([species] + coords)] = self._model.der_bulk_free_energy(species, rho_species) - 2 * self._k_laplacian * self._cell_laplacian(self._rho, species, coords)

        # Integrate time derivatives
        for idx in range(self._N_bins):
            coords = self._fill_coords(idx)
            for species in range(self._model.N_species):
                self._rho[tuple([species] + coords)] += self._M * self._cell_laplacian(rho_der, species, coords) * self._dt

    def _fill_coords(self, idx):
        coords = []
        for d in range(self._dim):
            coords.append(idx & self._N_per_dim_minus_one)
            idx >>= self._log2_N_per_dim
        return coords

    def _cell_idx(self, coords):
        idx = 0
        multiply_by = 1
        for d in range(self._dim):
            idx += coords[d] * multiply_by
            multiply_by <<= self._log2_N_per_dim
        return idx

    def _cell_laplacian(self, field, species, coords):
        if self._dim == 1:
            idx_m = (coords[0] - 1 + self._N_bins) & self._N_per_dim_minus_one
            idx_p = (coords[0] + 1) & self._N_per_dim_minus_one
            return (field[species, idx_m] + field[species, idx_p] - 2.0 * field[species, coords[0]]) / (
                        self._dx * self._dx)
        elif self._dim == 2:
            coords_xmy = [(coords[0] - 1 + self._N_bins) & self._N_per_dim_minus_one, coords[1]]
            coords_xym = [coords[0], (coords[1] - 1 + self._N_bins) & self._N_per_dim_minus_one]
            coords_xpy = [(coords[0] + 1) & self._N_per_dim_minus_one, coords[1]]
            coords_xyp = [coords[0], (coords[1] + 1) & self._N_per_dim_minus_one]
            return (field[species, coords_xmy[0], coords_xmy[1]] + field[species, coords_xpy[0], coords_xpy[1]] +
                    field[species, coords_xym[0], coords_xym[1]] + field[species, coords_xyp[0], coords_xyp[1]] -
                    4 * field[species, coords[0], coords[1]]) / (self._dx * self._dx)
        # Add 3D implementation if needed
        else:
            raise NotImplementedError("cell_laplacian not implemented for this dimension")

    def laplacian(self, u):
        # keep general laplacian for now.
        lap = np.zeros_like(u)
        slices = [slice(1, -1)] * self._dim
        inner_u = u[tuple(slices)]

        for i in range(self._dim):
            shifted_plus = \
            np.pad(u, pad_width=tuple((0, 0) if j != i else (1, 0) for j in range(self._dim)), mode='constant')[
                tuple(s if j != i else slice(2, None) for j, s in enumerate(slices))]
            shifted_minus = \
            np.pad(u, pad_width=tuple((0, 0) if j != i else (0, 1) for j in range(self._dim)), mode='constant')[
                tuple(s if j != i else slice(0, -2) for j, s in enumerate(slices))]
            lap[tuple(slices)] += (shifted_plus - 2 * inner_u + shifted_minus) / (self._dx * self._dx)
        return lap