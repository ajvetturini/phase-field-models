"""
Simple PINN which uses a Landau Free Energy functional to solve the PDE
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import numpy as np

def laplacian(func, x, y, t):
    """Computes the Laplacian of a function with respect to x and y for batched inputs."""
    def partial_x(x_i, y_i, t_i):
        return jnp.squeeze(jax.grad(lambda _x: jnp.squeeze(func(_x, y_i, t_i)))(x_i))

    def partial2_x2_single(x_i, y_i, t_i):
        return jax.grad(partial_x, argnums=0)(x_i, y_i, t_i)
    d2f_dx2 = jax.vmap(partial2_x2_single)(x, y, t)

    def partial_y(x_i, y_i, t_i):
        return jnp.squeeze(jax.grad(lambda _y: jnp.squeeze(func(x_i, _y, t_i)))(y_i))

    def partial2_y2_single(x_i, y_i, t_i):
        return jax.grad(partial_y, argnums=1)(x_i, y_i, t_i)
    d2f_dy2 = jax.vmap(partial2_y2_single)(x, y, t)

    return d2f_dx2 + d2f_dy2

def df_drho_scalar(rho_scalar):
    return jax.jit(jax.grad(lambda r: -0.5 * 0.9 * r**2 + 0.25 * r**4))(rho_scalar)

def cahn_hilliard_pinn(rho_pred_func, x, y, t, beta, K, M_prime):
    """
    Computes the residual of the PDE using JAX.

    Args:
        rho_pred_func (callable): A function that predicts the density rho at (x, y, t).
        x (jax.numpy.ndarray): The spatial x-coordinates (N_collocation, 1).
        y (jax.numpy.ndarray): The spatial y-coordinates (N_collocation, 1).
        t (jax.numpy.ndarray): The time coordinates (N_collocation, 1).
        beta (jax.numpy.ndarray): The parameter beta.
        K (jax.numpy.ndarray): The parameter K.
        M_prime (jax.numpy.ndarray): The parameter M'.

    Returns:
        jax.numpy.ndarray: The residual of the PDE (N_collocation, 1).
    """

    # Compute drho/dt for each collocation point
    def drho_dt_single(x_i, y_i, t_i):
        return jax.grad(
            lambda _t: jnp.squeeze(rho_pred_func(x_i, y_i, t_i)))(t_i)

    drho_dt = jax.vmap(drho_dt_single)(x, y, t)

    laplacian_df_drho = laplacian(lambda _x, _y, _t: df_drho_scalar(jnp.squeeze(rho_pred_func(_x, _y, _t))), x, y, t)

    # Compute the biharmonic term (Laplacian of Laplacian of rho)
    laplacian_laplacian_rho = laplacian(rho_pred_func, x, y, t)

    pde = drho_dt - M_prime * (beta * laplacian_df_drho - beta * K * laplacian_laplacian_rho)
    return pde

class Net2D(nn.Module):
    @nn.compact
    def __call__(self, x, y, t):
        inputs = jnp.concatenate([x, y, t], axis=-1)
        hidden = nn.Dense(features=20)(inputs)
        hidden = nn.tanh(hidden)
        hidden = nn.Dense(features=20)(hidden)
        hidden = nn.tanh(hidden)
        hidden = nn.Dense(features=20)(hidden)
        hidden = nn.tanh(hidden)
        hidden = nn.Dense(features=20)(hidden)
        hidden = nn.tanh(hidden)
        output = nn.Dense(features=1)(hidden)
        return output

def train_step(state, batch, beta, K, M_prime):
    def loss_fn(params):
        rho_pred_init = state.apply_fn({'params': params}, batch['x_init'], batch['y_init'], batch['t_init'])
        loss_initial = jnp.mean((rho_pred_init - batch['rho_init'])**2)

        def rho_pred_col(x, y, t):
            return state.apply_fn({'params': params}, x, y, t)

        pde_res = cahn_hilliard_pinn(rho_pred_col, batch['x_col'], batch['y_col'], batch['t_col'],
                                     beta, K, M_prime)
        loss_pde = jnp.mean(pde_res**2)
        loss = loss_initial + loss_pde
        return loss, loss_initial

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_initial), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = {'loss': loss, 'loss_initial': loss_initial, 'loss_pde': loss - loss_initial}
    return new_state, metrics

def train_pinn_single_species_2d_jax(initial_rho_jax, spatial_domain, time_domain,
                                      beta_true, K_true, M_prime_true,
                                      num_epochs=10000, learning_rate=1e-3, num_collocation_points=10000,
                                      num_initial_points=5000, batch_size=None):
    """
    Trains a PINN using Flax/JAX to solve the 2D PDE for a single species and learns the parameters.

    Args:
        initial_rho_jax (jax.numpy.ndarray): Initial density field at t=0 (shape: [num_x, num_y]).
        spatial_domain (tuple): Tuple (x_min, x_max, y_min, y_max) defining the spatial domain.
        time_domain (tuple): Tuple (t_min, t_max) defining the time domain.
        beta_true (float): The true value of beta.
        K_true (float): The true value of K.
        M_prime_true (float): The true value of M'.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        num_collocation_points (int): Number of collocation points in the domain.
        num_initial_points (int): Number of points to enforce the initial condition.
        batch_size (int, optional): Batch size for training. If None, uses all points at once.

    Returns:
        tuple: (trained_state, history of metrics, learned_params)
    """
    key = jax.random.PRNGKey(0)
    model = Net2D()

    # Initialize parameters
    dummy_input = jnp.ones((1, 3))  # (x, y, t)
    params = model.init(key, dummy_input[:, 0:1], dummy_input[:, 1:2], dummy_input[:, 2:3])['params']
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Define learnable parameters
    beta = jnp.array(beta_true, dtype=jnp.float32)
    K = jnp.array(K_true, dtype=jnp.float32)
    M_prime = jnp.array(M_prime_true, dtype=jnp.float32)

    # Prepare initial condition data
    x_min, x_max, y_min, y_max = spatial_domain
    t_min, t_max = time_domain
    num_x, num_y = initial_rho_jax.shape
    x_init_np = jnp.linspace(x_min, x_max, num_x)
    y_init_np = jnp.linspace(y_min, y_max, num_y)
    X_init_np, Y_init_np = jnp.meshgrid(x_init_np, y_init_np)
    T_init_np = jnp.zeros_like(X_init_np)
    rho_init_flat = initial_rho_jax.flatten().reshape(-1, 1)

    x_init = X_init_np.flatten().reshape(-1, 1)
    y_init = Y_init_np.flatten().reshape(-1, 1)
    t_init = T_init_np.flatten().reshape(-1, 1)
    rho_init = rho_init_flat

    # Generate collocation points
    key, subkey = jax.random.split(key)
    x_col = jax.random.uniform(subkey, (num_collocation_points, 1), minval=x_min, maxval=x_max)
    key, subkey = jax.random.split(key)
    y_col = jax.random.uniform(subkey, (num_collocation_points, 1), minval=y_min, maxval=y_max)
    key, subkey = jax.random.split(key)
    t_col = jax.random.uniform(subkey, (num_collocation_points, 1), minval=t_min, maxval=t_max)

    batch = {
        'x_init': x_init,
        'y_init': y_init,
        't_init': t_init,
        'rho_init': rho_init,
        'x_col': x_col,
        'y_col': y_col,
        't_col': t_col,
    }

    if batch_size is not None:
        num_batches = (num_initial_points + num_collocation_points) // batch_size
        # Implement batching logic here if needed
        raise NotImplementedError("Batching is not yet implemented in this JAX version.")

    metrics_history = []

    compiled_train_step = jax.jit(train_step)

    for epoch in range(num_epochs):
        state, metrics = compiled_train_step(state, batch, beta, K, M_prime)
        metrics_history.append(metrics)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Loss_initial: {metrics['loss_initial']:.4f}, "
                  f"Loss_pde: {metrics['loss_pde']:.4f}")

    learned_params = {"beta": float(beta), "K": float(K), "M_prime": float(M_prime)}
    return state, metrics_history, learned_params


def export_results(output_filepath, ic, trained_state, metrics_history, learned_params):
    """ Pickles + Writes results of training """
    # Print Results:
    print("\nLearned Parameters:")                  # Note: We can use the PINN to learn optimal parameters
    print(f"Beta: {learned_params['beta']:.4f}")
    print(f"K: {learned_params['K']:.4f}")
    print(f"M': {learned_params['M_prime']:.4f}")

    # Pickle + store results s.t. I can read them in later:
    # Pickle and store results
    results = {
        'trained_state': trained_state,
        'initial_condition': ic,
        'metrics_history': metrics_history,
        'learned_params': learned_params
    }
    try:
        with open(output_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults successfully exported to: {output_filepath}")
    except Exception as e:
        print(f"\nError during pickling: {e}")

    # Plot the loss
    losses = [m['loss'] for m in metrics_history]
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    # NOTE: These values are from jax_input in Landau
    N = 64
    spatial_domain = (0.0, float(N), 0.0, float(N))  # dx (default = 1) * N (where N is # grid elements)
    time_domain = (0.0, 1000.0)  # num_steps * dt

    # Create initial density, I am just copying from phase_field_model.py init from initial_density of 0.01
    densities = jnp.array([float(0.01)] * 1)
    x, y = jnp.meshgrid(jnp.arange(N), jnp.arange(N))
    r = jnp.sqrt(x ** 2 + y ** 2)
    k = 0  # Default 0
    modulation = 1e-2 * jnp.cos(k * r)
    prng, subkey = jax.random.split(jax.random.PRNGKey(8))
    noise = jax.random.uniform(subkey, shape=(1, N, N))
    random_factor = noise - 0.5
    initial_rho_jax = densities[:, None, None] * (1.0 + 2.0 * modulation[None, :, :] * random_factor)
    initial_rho_jax = initial_rho_jax[0]  # CURRENTLY: PINN only looking at 1-species (focus on multi-species later)
    init_rho = deepcopy(np.array(initial_rho_jax))

    # Define the true parameters
    # Note that these presume periodic boundary conditions inherently (and embed a 2D stencil of NxN grid where N = 2^n)
    beta_true = 1.0     # Scalar, 1.0 in this case
    K_true = 2.0        # This corresponds to self._interface_scalar * self._k_laplacian (2.0 * 1.0 for Landau)
    M_prime = 1.0  # This corresponds to self._M (1.0 for Landau)

    # Train the PINN
    trained_state, metrics_history, learned_params = train_pinn_single_species_2d_jax(
        initial_rho_jax, spatial_domain, time_domain,
        beta_true, K_true, M_prime,
        num_epochs=10, learning_rate=1e-3, num_collocation_points=15000,  # Just testing right now
        num_initial_points=N * N
    )

    export_results(r'./landau_pinn_test.pkl', init_rho, trained_state, metrics_history, learned_params)
