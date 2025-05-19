import os
import time
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from flax import linen as nn
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import orbax.checkpoint
from typing import Sequence
from functools import partial

# Define Constants & Geometry:
T_start = 0.0
T_step = 0.25
T_end = 0.25
WIDTH = LENGTH = 1.0
start_width = start_length = -1.0  # Domain in X / Y goes from [-1, 1]

# Scaling constants:
epsilon = 0.05  # Interfacial width parameter
r = 0.4

class PINN(nn.Module):
    # Standard feed-forward NN w/ tanh activation function
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # x has shape (batch_size, 3) for (x, y, t)
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        return x

def create_model(input_dim=3, hidden_layers=5, neurons_per_layer=128, output_dim=2):
    # 3 input parameters: X, Y, t
    # 2 output parameters: u, mu (concentration & chemical potential)
    features = [input_dim] + [neurons_per_layer] * hidden_layers + [output_dim]
    return PINN(features=features)

def create_train_state(model, rng, learning_rate=1e-3):
    """Creates initial `TrainState` for model."""
    params = model.init(rng, jnp.ones((1, 3)))  # Initialize with dummy input
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@partial(jax.jit, static_argnums=(1,))
def compute_pde_residuals(params, apply_fn, x_domain):
    """Compute the PDE residuals for the Cahn-Hilliard equation."""

    def net_fn(x):
        return apply_fn(params, x)

    # Automatic differentiation for computing derivatives
    def compute_derivatives(x):
        # Get model predictions
        y_pred = net_fn(x)
        u = y_pred[:, 0:1]
        mu = y_pred[:, 1:2]

        # Create a function that wraps the NN and can be differentiated
        # This function takes separate x, y, t inputs for easier differentiation
        def u_fn(x_val, y_val, t_val):
            x_single = jnp.array([x_val, y_val, t_val])
            return net_fn(x_single.reshape(1, -1))[0, 0]

        def mu_fn(x_val, y_val, t_val):
            x_single = jnp.array([x_val, y_val, t_val])
            return net_fn(x_single.reshape(1, -1))[0, 1]

        # Define derivative functions for time and space:
        du_dt_fn = grad(u_fn, argnums=2)

        # For spatial derivatives we need to handle the 2D case properly
        # Define specific derivative functions for each second derivative
        def u_xx(x_val, y_val, t_val):
            return grad(grad(u_fn, argnums=0), argnums=0)(x_val, y_val, t_val)

        def u_yy(x_val, y_val, t_val):
            return grad(grad(u_fn, argnums=1), argnums=1)(x_val, y_val, t_val)

        def mu_xx(x_val, y_val, t_val):
            return grad(grad(mu_fn, argnums=0), argnums=0)(x_val, y_val, t_val)

        def mu_yy(x_val, y_val, t_val):
            return grad(grad(mu_fn, argnums=1), argnums=1)(x_val, y_val, t_val)

        # Vectorize computations for batch processing
        # For time derivatives, we need to unpack the input points
        def compute_du_dt(point):
            x_val, y_val, t_val = point
            return du_dt_fn(x_val, y_val, t_val)

        def compute_u_xx(point):
            x_val, y_val, t_val = point
            return u_xx(x_val, y_val, t_val)

        def compute_u_yy(point):
            x_val, y_val, t_val = point
            return u_yy(x_val, y_val, t_val)

        def compute_mu_xx(point):
            x_val, y_val, t_val = point
            return mu_xx(x_val, y_val, t_val)

        def compute_mu_yy(point):
            x_val, y_val, t_val = point
            return mu_yy(x_val, y_val, t_val)

        # Apply vectorized operations to all points
        du_dt_vals = vmap(compute_du_dt)(x).reshape(-1, 1)  # Shape (30000, 1)
        du_dxx = vmap(compute_u_xx)(x).reshape(-1, 1)  # Shape (30000, 1)
        du_dyy = vmap(compute_u_yy)(x).reshape(-1, 1)  # Shape (30000, 1)
        dmu_dxx = vmap(compute_mu_xx)(x).reshape(-1, 1)  # Shape (30000, 1)
        dmu_dyy = vmap(compute_mu_yy)(x).reshape(-1, 1)  # Shape (30000, 1)

        # Compute bulk energy (double-well potential)
        f_h = u ** 3 - u

        # PDE residuals
        eq1 = du_dt_vals - (dmu_dxx + dmu_dyy)
        eq2 = mu - (f_h - epsilon ** 2 * (du_dxx + du_dyy))

        return eq1, eq2, u, mu

    return compute_derivatives(x_domain)


def enforce_periodic_bc(params, apply_fn, x_boundary_pairs):
    """Enforce periodic boundary conditions.
    x_boundary_pairs contains pairs of points (x1, x2) that should have the same value.
    """
    def net_fn(x):
        return apply_fn(params, x)

    # Vectorize prediction function
    vmap_net_fn = vmap(lambda x: net_fn(x.reshape(1, -1)))

    # Split boundary pairs into left/bottom and right/top
    x_left = x_boundary_pairs[:, 0, :]  # Points on left/bottom boundary
    x_right = x_boundary_pairs[:, 1, :]  # Corresponding points on right/top boundary

    # Get predictions for both sets of points
    y_left = vmap_net_fn(x_left)
    y_right = vmap_net_fn(x_right)

    # Value continuity residual
    residual_values = y_left - y_right

    return residual_values


def enforce_derivative_periodic_bc(params, apply_fn, x_boundary_pairs):
    """Enforce periodic boundary conditions on derivatives.
    x_boundary_pairs contains pairs of points (x1, x2) that should have the same derivative value.
    """

    def net_fn(x):
        return apply_fn(params, x)

    # Define separate u and mu functions for cleaner differentiation
    def u_fn(x_val, y_val, t_val):
        x_single = jnp.array([x_val, y_val, t_val])
        return net_fn(x_single.reshape(1, -1))[0, 0]

    def mu_fn(x_val, y_val, t_val):
        x_single = jnp.array([x_val, y_val, t_val])
        return net_fn(x_single.reshape(1, -1))[0, 1]

    # Define gradient functions for u with respect to x and y
    du_dx_fn = grad(u_fn, argnums=0)
    du_dy_fn = grad(u_fn, argnums=1)

    # Define gradient functions for mu with respect to x and y
    dmu_dx_fn = grad(mu_fn, argnums=0)
    dmu_dy_fn = grad(mu_fn, argnums=1)

    # Vectorize computations for boundary points
    def compute_all_derivatives(point):
        x_val, y_val, t_val = point
        return (
            du_dx_fn(x_val, y_val, t_val),
            du_dy_fn(x_val, y_val, t_val),
            dmu_dx_fn(x_val, y_val, t_val),
            dmu_dy_fn(x_val, y_val, t_val)
        )

    # Split boundary pairs into left/bottom and right/top
    x_left = x_boundary_pairs[:, 0, :]  # Points on left/bottom boundary
    x_right = x_boundary_pairs[:, 1, :]  # Corresponding points on right/top boundary

    if x_left.shape[0] > 0:  # Ensure we have points to process
        # Compute all derivatives at the boundary points
        du_dx_left, du_dy_left, dmu_dx_left, dmu_dy_left = vmap(compute_all_derivatives)(x_left)
        du_dx_right, du_dy_right, dmu_dx_right, dmu_dy_right = vmap(compute_all_derivatives)(x_right)

        # Compute derivative residuals
        residual_du_dx = du_dx_left - du_dx_right
        residual_du_dy = du_dy_left - du_dy_right
        residual_dmu_dx = dmu_dx_left - dmu_dx_right
        residual_dmu_dy = dmu_dy_left - dmu_dy_right

        # Stack all residuals into a single array
        return jnp.stack([
            residual_du_dx,
            residual_du_dy,
            residual_dmu_dx,
            residual_dmu_dy
        ], axis=1)
    else:
        # Return empty array if no points
        return jnp.zeros((0, 4))

def enforce_initial_condition(params, apply_fn, x_ic, ic_values):
    """Enforce initial conditions."""
    # Predict at initial condition points
    y_pred = apply_fn(params, x_ic)
    u_pred = y_pred[:, 0:1]

    # Compute residual between prediction and actual IC
    residual_ic = u_pred - ic_values

    return residual_ic


def compute_loss(params, apply_fn, x_domain, x_boundary_pairs_x, x_boundary_pairs_y,
                 x_ic, ic_values, loss_weights):
    """Compute the total loss considering PDE residuals, BCs, and ICs."""
    # PDE residuals
    eq1, eq2, _, _ = compute_pde_residuals(params, apply_fn, x_domain)

    # Boundary conditions
    bc_residual_x = enforce_periodic_bc(params, apply_fn, x_boundary_pairs_x)
    bc_residual_y = enforce_periodic_bc(params, apply_fn, x_boundary_pairs_y)

    # Derivative boundary conditions - only if we have boundary points
    if x_boundary_pairs_x.shape[0] > 0 and x_boundary_pairs_y.shape[0] > 0:
        deriv_bc_residual_x = enforce_derivative_periodic_bc(params, apply_fn, x_boundary_pairs_x)
        deriv_bc_residual_y = enforce_derivative_periodic_bc(params, apply_fn, x_boundary_pairs_y)
    else:
        # Create empty arrays with correct dimension if no boundary points
        deriv_bc_residual_x = jnp.zeros((0, 4))
        deriv_bc_residual_y = jnp.zeros((0, 4))

    # Initial conditions
    ic_residual = enforce_initial_condition(params, apply_fn, x_ic, ic_values)

    # Compute mean squared errors
    mse_eq1 = jnp.mean(eq1 ** 2)
    mse_eq2 = jnp.mean(eq2 ** 2)

    mse_bc_x = jnp.mean(bc_residual_x ** 2) if bc_residual_x.size > 0 else jnp.zeros(1)
    mse_bc_y = jnp.mean(bc_residual_y ** 2) if bc_residual_y.size > 0 else jnp.zeros(1)

    # Only compute derivative BC losses if we have points
    if deriv_bc_residual_x.size > 0:
        mse_deriv_bc_x = jnp.mean(deriv_bc_residual_x ** 2)
    else:
        mse_deriv_bc_x = jnp.zeros(1)

    if deriv_bc_residual_y.size > 0:
        mse_deriv_bc_y = jnp.mean(deriv_bc_residual_y ** 2)
    else:
        mse_deriv_bc_y = jnp.zeros(1)

    mse_ic = jnp.mean(ic_residual ** 2) if ic_residual.size > 0 else jnp.zeros(1)

    # Combine losses with weights
    w_eq1, w_eq2, w_bc_x, w_bc_y, w_deriv_bc_x, w_deriv_bc_y, w_ic = loss_weights

    total_loss = (
            w_eq1 * mse_eq1 +
            w_eq2 * mse_eq2 +
            w_bc_x * mse_bc_x +
            w_bc_y * mse_bc_y +
            w_deriv_bc_x * mse_deriv_bc_x +
            w_deriv_bc_y * mse_deriv_bc_y +
            w_ic * mse_ic
    )

    component_losses = {
        "eq1": w_eq1 * mse_eq1,
        "eq2": w_eq2 * mse_eq2,
        "bc_x": w_bc_x * mse_bc_x,
        "bc_y": w_bc_y * mse_bc_y,
        "deriv_bc_x": w_deriv_bc_x * mse_deriv_bc_x,
        "deriv_bc_y": w_deriv_bc_y * mse_deriv_bc_y,
        "ic": w_ic * mse_ic,
        "total": total_loss
    }

    return total_loss, component_losses


@jit
def train_step(state, x_domain, x_boundary_pairs_x, x_boundary_pairs_y,
               x_ic, ic_values, loss_weights):
    """Perform a single training step."""

    def loss_fn(params):
        loss, component_losses = compute_loss(
            params, state.apply_fn, x_domain,
            x_boundary_pairs_x, x_boundary_pairs_y,
            x_ic, ic_values, loss_weights
        )
        return loss, component_losses

    (loss, component_losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, component_losses

def initial_condition(x):
    x_vals = x[:, 0:1]
    y_vals = x[:, 1:2]

    # Create smoothened areas as initial condition
    R1 = jnp.sqrt((x_vals - 0.7 * r) ** 2 + y_vals ** 2)
    R2 = jnp.sqrt((x_vals + 0.7 * r) ** 2 + y_vals ** 2)
    phi_R1 = jnp.tanh((r - R1) / (2 * epsilon))
    phi_R2 = jnp.tanh((r - R2) / (2 * epsilon))
    return jnp.maximum(phi_R1, phi_R2)

def create_initial_condition_2d(predictions, x_range, y_range):
    if predictions.size != len(x_range) * len(y_range):
        raise ValueError(f"Cannot reshape array of size {predictions.size} into shape ({len(x_range)}, {len(y_range)})")
    predictions_reshaped = predictions.reshape(len(x_range), len(y_range))

    # Use SciPy grid interpolator to define a function of the same size as initial_condition() above
    interp_func = RegularGridInterpolator((x_range, y_range), predictions_reshaped)
    return lambda X: interp_func(X[:, :2]).reshape(-1, 1)


def sample_points(rng, num_domain, num_boundary, num_initial, current_time, next_time):
    """Sample points for training."""
    keys = jax.random.split(rng, 10)

    # Domain points
    x_domain = jax.random.uniform(keys[0], shape=(num_domain, 1), minval=start_width, maxval=WIDTH)
    y_domain = jax.random.uniform(keys[1], shape=(num_domain, 1), minval=start_length, maxval=LENGTH)
    t_domain = jax.random.uniform(keys[2], shape=(num_domain, 1), minval=current_time, maxval=next_time)
    domain_points = jnp.hstack([x_domain, y_domain, t_domain])

    # Boundary points for x direction (left-right)
    num_side_points = num_boundary // 4
    x_boundary_left = jnp.full((num_side_points, 1), start_width)
    x_boundary_right = jnp.full((num_side_points, 1), WIDTH)
    y_boundary_x = jax.random.uniform(keys[3], shape=(num_side_points, 1), minval=start_length, maxval=LENGTH)
    t_boundary_x = jax.random.uniform(keys[4], shape=(num_side_points, 1), minval=current_time, maxval=next_time)

    # Create pairs of points for periodic BCs in x direction
    left_points = jnp.hstack([x_boundary_left, y_boundary_x, t_boundary_x])
    right_points = jnp.hstack([x_boundary_right, y_boundary_x, t_boundary_x])
    x_boundary_pairs_x = jnp.stack([left_points, right_points], axis=1)

    # Boundary points for y direction (bottom-top)
    y_boundary_bottom = jnp.full((num_side_points, 1), start_length)
    y_boundary_top = jnp.full((num_side_points, 1), LENGTH)
    x_boundary_y = jax.random.uniform(keys[5], shape=(num_side_points, 1), minval=start_width, maxval=WIDTH)
    t_boundary_y = jax.random.uniform(keys[6], shape=(num_side_points, 1), minval=current_time, maxval=next_time)

    # Create pairs of points for periodic BCs in y direction
    bottom_points = jnp.hstack([x_boundary_y, y_boundary_bottom, t_boundary_y])
    top_points = jnp.hstack([x_boundary_y, y_boundary_top, t_boundary_y])
    x_boundary_pairs_y = jnp.stack([bottom_points, top_points], axis=1)

    #key_ic = jax.random.split(keys[0], 3)
    x_ic = jax.random.uniform(keys[7], shape=(num_initial, 1), minval=start_width, maxval=WIDTH)
    y_ic = jax.random.uniform(keys[8], shape=(num_initial, 1), minval=start_length, maxval=LENGTH)
    t_ic = jnp.full((num_initial, 1), current_time)
    ic_points = jnp.hstack([x_ic, y_ic, t_ic])

    return keys[9], domain_points, x_boundary_pairs_x, x_boundary_pairs_y, ic_points

def predict_at_time_2d(model, params, time, Nx, Ny):
    """Use the model to predict system at time `time`"""
    x_lins = jnp.linspace(start_width, WIDTH, Nx)
    y_lins = jnp.linspace(start_length, LENGTH, Ny)
    t_array = jnp.array([time])

    # Create grid
    xx, yy, tt = jnp.meshgrid(x_lins, y_lins, t_array)
    test_domain = jnp.vstack((xx.flatten(), yy.flatten(), tt.flatten())).T

    # Convert to JAX array
    test_domain_jax = jnp.array(test_domain)

    # Predict
    prediction = model.apply(params, test_domain_jax)

    # Reshape for visualization
    u_pred = prediction[:, 0].reshape(Ny, Nx)
    mu_pred = prediction[:, 1].reshape(Ny, Nx)

    return u_pred, mu_pred, x_lins, y_lins

def plot_solution(u_pred, mu_pred, x_lins, y_lins, time):
    """Plot the predicted solution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot u (order parameter)
    im1 = ax1.contourf(x_lins, y_lins, u_pred, levels=50, cmap='viridis')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(f'Order Parameter $u(x,y,t={time:.2f})$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Plot mu (chemical potential)
    im2 = ax2.contourf(x_lins, y_lins, mu_pred, levels=50, cmap='plasma')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(f'Chemical Potential $\mu(x,y,t={time:.2f})$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.tight_layout()
    return fig

def save_checkpoint(state, step, checkpoint_dir):
    """Save model checkpoint."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(
        f"{checkpoint_dir}/checkpoint_{step}",
        state,
        force=True
    )

def load_checkpoint(state, step, checkpoint_dir):
    """Load model checkpoint."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore(
        f"{checkpoint_dir}/checkpoint_{step}",
        state
    )
    return state

def train_cahn_hilliard_pinn():
    # Hyperparameters
    num_domain = 30000
    num_boundary = 1600
    num_initial = 4096
    loss_weights = [1, 1, 1, 1, 1, 1, 1000]  # [eq1, eq2, bc_x, bc_y, deriv_bc_x, deriv_bc_y, ic]
    training_iterations = 50000
    learning_rate = 1e-3
    checkpoint_interval = 1000
    plot_interval = 2000
    N_square = 70  # Grid size for visualization

    # Setup directory for checkpoints and plots
    base_save_dir = './jax_pinn_results'
    checkpoint_dir = f"{base_save_dir}/checkpoints"
    plot_dir = f"{base_save_dir}/plots"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Initialize model and training state
    model = create_model()
    rng = jax.random.PRNGKey(3)
    state = create_train_state(model, rng, learning_rate)

    # Time-marching scheme
    sample_key = jax.random.PRNGKey(8)
    current_time = T_start
    models = []
    predictions = []
    trajectory = []

    while current_time < T_end:
        next_time = current_time + T_step
        print(f"Training for time interval [{current_time}, {next_time}]")

        # Sample points for this time interval
        sample_key, domain_points, x_boundary_pairs_x, x_boundary_pairs_y, ic_points = sample_points(
            sample_key, num_domain, num_boundary, num_initial, current_time, next_time
        )

        # Prepare initial condition values
        if current_time == 0:
            ic_function = initial_condition
        else:
            # Create IC from previous prediction
            x_linspace = jnp.linspace(start_width, WIDTH, N_square)
            y_linspace = jnp.linspace(start_length, LENGTH, N_square)
            previous_prediction = predictions[-1][0].reshape(N_square, N_square)
            ic_function = create_initial_condition_2d(previous_prediction, x_linspace, y_linspace)

        ic_values = ic_function(ic_points)

        # Convert numpy arrays to JAX arrays
        domain_points_jax = jnp.array(domain_points)
        x_boundary_pairs_x_jax = jnp.array(x_boundary_pairs_x)
        x_boundary_pairs_y_jax = jnp.array(x_boundary_pairs_y)
        ic_points_jax = jnp.array(ic_points)
        ic_values_jax = jnp.array(ic_values)

        # Training loop
        for step in range(training_iterations):
            start_time = time.time()

            # Perform training step
            state, loss, component_losses = train_step(
                state, domain_points_jax, x_boundary_pairs_x_jax, x_boundary_pairs_y_jax,
                ic_points_jax, ic_values_jax, loss_weights
            )

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss}, Time: {time.time() - start_time:.4f}s")
                # Print component losses
                for k, v in component_losses.items():
                    print(f"  {k}: {v}")

            # Save checkpoint
            if step % checkpoint_interval == 0:
                save_checkpoint(state, step, checkpoint_dir)

            # Generate and save plot
            if step % plot_interval == 0:
                u_pred, mu_pred, x_lins, y_lins = predict_at_time_2d(
                    model, state.params, (current_time + next_time) / 2, N_square, N_square
                )
                # NOTE: Return back image data and write to trajectory..
                trajectory.append([current_time, u_pred, mu_pred, x_lins, y_lins])

        # After training, predict at the end of the time interval
        u_pred, mu_pred, x_lins, y_lins = predict_at_time_2d(
            model, state.params, next_time, N_square, N_square
        )

        # Save final prediction for this time step
        predictions.append((u_pred, mu_pred))
        models.append(state)

        ## TODO: Add in adaptive sampling here to further minimize residuals:
        """
        for i in range(num_adaptive_iteration_steps):
        current_total = len(model.data.train_x)  # current total points
        if current_total >= max_total_points:
            print(f"Maximum point limit reached, ending adaptive sampling for time={current_time}")
            break

        # Adaptive sampling
        x_new = adaptive_sampling(model, geomtime, cahn_hilliard, num_samples=adaptive_samples, 
                                  error_threshold=error_threshold, neighborhood_size=neighborhood_size)

        # Add new points to the training data
        if len(x_new) > 0:
            print(f"Adaptive iteration {number_of_itrations+1}: Adding {len(x_new)} new points")
            model.data.add_anchors(x_new)
        else:
            print(f"Adaptive iteration {number_of_itrations+1}: No new points added")

        # Continually train the model
        model.compile(adaptive_optimizer, lr=adaptive_learning_rate, loss=adaptive_loss, loss_weights=adaptive_weights)
        model.train(iterations=adaptive_iterations, batch_size=batch_size, callbacks=[early_stopping])
        """

        # Move to next time interval
        current_time = next_time

    return models, predictions


if __name__ == "__main__":
    models, predictions = train_cahn_hilliard_pinn()

    print("Training completed successfully!")
