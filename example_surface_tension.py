import numpy as np
from pfm.energy_models import SimpleWertheim
from pfm.phase_field_models import CahnHilliard
from pfm.integrators import ExplicitEuler
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import toml
cfg = r'C:\Users\Anthony\Documents\GitHub\phase-field-models\Examples\Wertheim\simple_wertheim\simple_wertheim_test.toml'
bulk = SimpleWertheim(toml.load(cfg))
integrator = ExplicitEuler(bulk, toml.load(cfg))
ch = CahnHilliard(bulk, toml.load(cfg), integrator, 1)
dx = toml.load(cfg).get('dx')
def compute_surface_tension(rho_grid, show_plot: bool = False):
    """ Estimate surface tension gamma from a 2D density field. """
    species_0 = rho_grid[0]
    bulk_array = jnp.array(rho_grid)
    F_coex = ch.average_free_energy(bulk_array) * (rho_grid.shape[1] * rho_grid.shape[2])  # Total Energy (scaled by bin

    # Determine equilibrium densities by averaging over bulk regions
    rho_min, rho_max = species_0.min(), species_0.max()
    threshold = 0.5 * (rho_min + rho_max)
    gas_mask = species_0 < threshold
    liquid_mask = species_0 >= threshold
    rho_gas_eq = jnp.mean(species_0[gas_mask])  # This might break in jit...
    rho_liquid_eq = jnp.mean(species_0[liquid_mask])

    # Plot gas_maksk and liquid_mask (0 / 1) and on the right plot the imshow of rho_grid
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gas_mask, origin='lower', cmap='Blues')
        axes[0].set_title('Gas mask')
        axes[0].axis('off')
        axes[1].imshow(liquid_mask, origin='lower', cmap='Reds')
        axes[1].set_title('Liquid mask')
        axes[1].axis('off')
        im = axes[2].imshow(rho_grid[0], origin='lower', cmap='viridis')
        axes[2].set_title('Density field (rho)')
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    # Create two homogenous systems
    gas_system = jnp.full_like(rho_grid, rho_gas_eq)  # System that is 100% gas at equilibrium density
    liquid_system = jnp.full_like(rho_grid, rho_liquid_eq)  # System that is 100% liquid at equilibrium density
    f_gas_density = ch.average_free_energy(gas_system)
    f_liquid_density = ch.average_free_energy(liquid_system)

    # Calculate the reference free energy based on the volume of each phase in the original coexisting system.
    N_gas_cells = jnp.sum(gas_mask)
    N_liquid_cells = jnp.sum(liquid_mask)

    F_reference = (f_gas_density * N_gas_cells) + (f_liquid_density * N_liquid_cells)

    # Estimate interface area using a mask
    dilated_liquid = binary_dilation(liquid_mask)
    interface_mask = dilated_liquid & gas_mask
    A_total = np.sum(interface_mask) * np.sqrt(dx * dx)

    # Surface tension
    gamma = (F_coex - F_reference) / (2 * A_total)

    return gamma, (F_coex, F_reference)


def compute_surface_tension_mechanical(rho_grid, kappa, show_plot: bool = False):
    """
    Estimate surface tension gamma using the direct integral of the interfacial energy density.
    This method is numerically stable and highly recommended for droplet systems.
    """
    # 1. Calculate the gradient of the density field for the first species
    # Assuming ch.gradient(rho) returns a tensor of shape (N_species, N_dim, Nx, Ny)
    grad_rho = ch.gradient(rho_grid)[0]  # Shape: (N_dim, Nx, Ny)

    # 2. Calculate the squared magnitude of the gradient at each point
    grad_rho_sq = jnp.sum(grad_rho ** 2, axis=0)  # Sum over the spatial dimensions (x, y)

    # 3. Integrate the interfacial energy density (kappa * |∇ρ|²) over the domain
    total_interfacial_energy = jnp.sum(kappa * grad_rho_sq) * dx * dx

    # 4. Calculate the total interface length (perimeter of all droplets)
    # Your method is a reasonable first approximation.
    species_0 = rho_grid[0]
    rho_min, rho_max = species_0.min(), species_0.max()
    threshold = 0.5 * (rho_min + rho_max)
    liquid_mask = species_0 >= threshold
    gas_mask = species_0 < threshold

    # Using scipy for a slightly more accurate perimeter estimation than pixel counting
    from scipy.ndimage import binary_erosion
    interface_mask = liquid_mask & ~binary_erosion(liquid_mask)
    L_total = jnp.sum(interface_mask) * dx  # Approximate total perimeter

    # For higher accuracy, consider using a contour-finding algorithm
    # from skimage.measure import find_contours, perimeter
    # contours = find_contours(liquid_mask, 0.5)
    # L_total = sum(perimeter(c, neighborhood=4) for c in contours) * dx

    # Avoid division by zero if no interface is present
    if L_total == 0:
        return 0.0

    # 5. Calculate surface tension
    gamma = total_interfacial_energy / L_total

    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot the original liquid mask
        axes[0].imshow(liquid_mask, cmap='Reds', origin='lower')
        axes[0].set_title('Original Liquid Mask')
        axes[0].axis('off')

        # Plot the interface identified via EROSION
        axes[1].imshow(interface_mask, cmap='gray', origin='lower')
        axes[1].set_title('Interface Mask (Erosion Method)')
        axes[1].axis('off')

        # Plot the interface identified via DILATION
        dilated_liquid = binary_dilation(liquid_mask)
        interface_mask2 = dilated_liquid & gas_mask
        axes[2].imshow(interface_mask2, cmap='gray', origin='lower')
        axes[2].set_title('Interface Mask (Dilation Method)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    return gamma


data_file = r"C:\Users\Anthony\Documents\GitHub\phase-field-models\Examples\Wertheim\simple_wertheim\cuda\last_0.dat"
data = np.loadtxt(data_file)
reshaped_array = data.reshape(1, 512, 512)
#surface_tension, components = compute_surface_tension(reshaped_array, True)
surface_tension = compute_surface_tension_mechanical(reshaped_array, 1e6, True)
print(surface_tension)