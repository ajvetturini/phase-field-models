""" Example of gradient-boosted Bayesian for inverse-design of epsilon parameter in Landau free energy model """
from pfm import SimulationManagerNoWrite, SimulationManager
from pfm.inverse_design import BOObjective, BOArgs, run_bayesian_optimization
import jax.numpy as jnp
import toml

def measure_characteristic_well_width(final_state: jnp.ndarray) -> jnp.ndarray:
    """ Differentiable evaluation of characteristic well width from final state for landau free energy model

    This takes in the final state from the numerical method simulation and computes the well width from which we are
    optimizing the epsilon parameter to achieve through Bayesian Optimization
    """
    grid = final_state - jnp.mean(final_state)  # Center about zero for power spectrum
    fft_grid = jnp.fft.fft2(grid)
    power_spectrum = jnp.abs(fft_grid) ** 2

    # Get wave numbers of grid points:
    N = grid.shape[0]
    k = jnp.fft.fftfreq(N) * N  # Multiply by N for pixel units
    kx, ky = jnp.meshgrid(k, k, indexing='ij')
    k_magnitude = jnp.sqrt(kx ** 2 + ky ** 2)

    # Calculate weighted average number (this is differentiable unlike argmax)
    total_power = jnp.sum(power_spectrum) + 1e-6
    k_avg = jnp.sum(k_magnitude * power_spectrum) / total_power

    # The characteristic length is 2*pi / k_avg and we add a small epsilon to avoid division by zero
    characteristic_length = (2.0 * jnp.pi) / (k_avg + 1e-6)
    return characteristic_length

class LandauBO(BOObjective):
    def __init__(self, toml_data, target_length_scale: float):
        if target_length_scale <= 0:
            raise ValueError("Target well width must be positive.")
        self.target_length_scale = target_length_scale
        super().__init__(toml_data)

    def forward_simulation(self, params, static_data) -> jnp.ndarray:
        """Implements the specific logic for the Landau well width calculation."""
        # First, update static_data with new epsilon selected by BO and run simulation
        epsilon = params['epsilon']
        safe_epsilon = jnp.maximum(epsilon, 1e-6)
        static_data['landau']['epsilon'] = safe_epsilon
        simulator = SimulationManagerNoWrite(static_data)

        # Run the simulation to obtain final state / order params for measuring output feature of interest
        # NOTE: Assumes static_data leads to valid final state (e.g., make sure num_steps is large enough!)
        final_state = simulator.run_system_no_logging()

        # Measure the max well width from the final state found and return
        final_state = final_state[0, :, :]  # Extract the single species order parameter since Landau is single-species
        characteristic_length = measure_characteristic_well_width(final_state)
        return characteristic_length

    def final_forward_simulation(self, params, static_data):
        """ This is a forward_simulation that WRITE the export using the input TOML data """
        epsilon = params['epsilon']
        safe_epsilon = jnp.maximum(epsilon, 1e-6)
        static_data['landau']['epsilon'] = float(safe_epsilon)  # You must manually update this!
        simulator = SimulationManager(static_data)
        final_state = simulator.run(override_use_jax=True)

        # Measure the max well width from the final state found and return
        final_state = final_state[0, :, :]  # Extract the single species order parameter since Landau is single-species
        characteristic_length = measure_characteristic_well_width(final_state)

        # Make sure you return the final evaluation AND the toml data
        return characteristic_length, static_data

    def loss_function(self, params, static_data) -> jnp.ndarray:
        """ Calculates loss based on the output of our forward_simulation, we seek to minimize this value """
        predicted_width = self.forward_simulation(params, static_data)
        return (predicted_width - self.target_length_scale) ** 2


if __name__ == "__main__":
    # INPUTS:
    TARGET_LENGTH_SCALE = 10.0      # Features to be 10 pixels wide on average is the inverse-design target
    TOML_DATA = toml.load(r'landau_input.toml')

    # Setup Bayesian Optimization objective and BOArgs:
    objective = LandauBO(TOML_DATA, target_length_scale=TARGET_LENGTH_SCALE)
    bo_args = BOArgs(
        write_location='',
        num_iterations=8,
        num_initial_points=4,
        batch_size=8,
        parameter_bounds={'epsilon': (0.1, 20.0)},
        verbose=True
    )
    carry = run_bayesian_optimization(objective, bo_args)
    best_loss, best_param = carry
    print(f'Optimal epsilon: {best_param["epsilon"]}')
