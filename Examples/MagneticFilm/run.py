from pfm import SimulationManager
import toml
import time
import numpy as np

def custom_initial_condition(init_phi):
    # init_phi is a zeros array of the simulation field shape
    shape = init_phi.shape
    num_species = shape[0]
    grid_shape = shape[1:]

    # Note that there is a species in the 0th idx
    for i in range(num_species):
        # Generate random values for the current species on the spatial grid
        init_phi[i] = 2 * np.random.rand(*grid_shape) - 1  # Values between -1 and 1

    return init_phi

c = toml.load('input_magnetic_film.toml')
manager = SimulationManager(c, custom_initial_condition=custom_initial_condition)  # Just pass the fn, no ()
start = time.time()
manager.run()
end = time.time() - start

minutes = int(end // 60)
seconds = int(end % 60)

print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')