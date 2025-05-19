from pfm import SimulationManager
import toml
import time
import numpy as np
from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)

class PINN_Test(FreeEnergyModel):
    def __init__(self, config):
        super().__init__(config)
        self._energy_config = config.get('PINN_test')
        self._delta = jnp.array(self._energy_config.get('delta', 1.0), dtype=jnp.float64)

        self._autograd_fn = jax.jit(jax.grad(self._elementwise_bulk_free_energy))

    def N_species(self):
        return 1

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, phi):
        """ Derivative of the double-well bulk free energy. """
        return jnp.pow(phi, 3) - phi

    def _elementwise_bulk_free_energy(self, phi_species):
        """ Calculates the double-well bulk free energy for each point in the grid. """
        return jnp.pow(phi_species, 4) - 2 * (jnp.pow(phi_species, 2))

    def _total_bulk_free_energy(self, rho_species):
        return jnp.sum(self._elementwise_bulk_free_energy(rho_species))

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy_autodiff(self, species, rho_species):
        """ Uses autodiff to evaluate the bulk_free_energy term """
        elementwise_grad_fn = jax.grad(self._total_bulk_free_energy)(rho_species)
        return elementwise_grad_fn

    def bulk_free_energy(self, rho_species):
        op = rho_species[0]  # Only 1 species
        return jnp.pow(op, 4) - 2 * jnp.pow(op, 2)


def custom_initial_condition(init_phi, r=0.4, epsilon=0.05, Lx=1.0, Ly=1.0):
    # init_phi is a zeros array of shape (num_species, Nx, Ny)
    num_species, Nx, Ny = init_phi.shape

    # Create meshgrid of coordinates scaled to [−Lx/2, Lx/2] and [−Ly/2, Ly/2]
    x = jnp.linspace(-Lx / 2, Lx / 2, Nx)
    y = jnp.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # Compute distance fields for two blobs
    R1 = jnp.sqrt((X - 0.7 * r)**2 + Y**2)
    R2 = jnp.sqrt((X + 0.7 * r)**2 + Y**2)

    # Compute smooth indicator functions using tanh
    phi_R1 = jnp.tanh((r - R1) / (2 * epsilon))
    phi_R2 = jnp.tanh((r - R2) / (2 * epsilon))
    blob_field = jnp.maximum(phi_R1, phi_R2)

    init_phi[0] = np.array(blob_field)

    # Verify via matplotlib
    """import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 4))
    plt.imshow(blob_field.T, origin='lower', extent=[-Lx, Lx, -Ly, Ly], cmap='viridis')
    plt.colorbar(label='phi')
    plt.title('Initial Condition (Species 0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()"""

    return init_phi

# NOTES
# Overall, this is a SHORT simulation! We are looking at the early separation (first few hundred timesteps)
# The system achieves a minimum relatively quickly (within few thousand steps), so if you run a system
# too long, the results file may not tell an accurate story.

# ALSO: If you specify a custom function using jax, you MUST specify the CUDA device here prior to importing jax
#       See how main.py is structured!
c = toml.load('input.toml')
manager = SimulationManager(c, custom_energy=PINN_Test, custom_initial_condition=custom_initial_condition)
start = time.time()
manager.run()
end = time.time() - start

minutes = int(end // 60)
seconds = int(end % 60)

print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')