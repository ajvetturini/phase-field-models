import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax.training import train_state as train_utils
from functools import partial
import time
from pfm.experimental.pinn_OLD_BACKUP_FAILING.networks import MLPFourier

# Define Landau free energy function
def landau_f(phi, epsilon=0.9):
    """ Landau Bulk Free Energy """
    return -0.5 * epsilon * phi ** 2 + 0.25 * phi ** 4


@jax.jit
def dfdphi(phi, epsilon=0.9):
    """ Derivative of Landau free energy """
    # TODO: Return this to autodiff at some point?
    return -epsilon * phi + phi ** 3


# Training configuration with learning rate schedule
def _create_train_state(model, learning_rate=1e-3):
    """Create initial training state with Adam optimizer"""
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))

    # Learning rate schedule: start high, then decay
    schedule_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=5000,
        alpha=0.1  # Don't decay to zero
    )

    # Optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Prevent exploding gradients
        optax.adam(learning_rate=schedule_fn)
    )

    return train_utils.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@partial(jax.jit, static_argnums=(1,))
def pinn_residual(params, apply_fn, coords):
    """
    More stable implementation of the Cahn-Hilliard PINN residual
    """

    def phi_fn(xyt):
        # Remove output tanh activation or use a different range
        phi = apply_fn(params, xyt).squeeze()
        return phi  # Allow φ to have a wider range than [-1, 1]

    # Time derivative
    def phi_t(xyt):
        return jax.grad(phi_fn, argnums=0)(xyt)[2]  # Derivative w.r.t. t (index 2)

    # Spatial derivatives for Laplacian
    def laplacian_phi(xyt):
        # Compute Hessian w.r.t spatial variables
        hess = jax.hessian(lambda xy_t: phi_fn(jnp.concatenate([xy_t[:2], xyt[2:3]])))(xyt[:2])
        return hess[0, 0] + hess[1, 1]  # ∂²φ/∂x² + ∂²φ/∂y²

    # Chemical potential
    def mu(xyt):
        phi = phi_fn(xyt)
        lap = laplacian_phi(xyt)
        kappa = 1.0
        return dfdphi(phi) - kappa * lap

    # Laplacian of chemical potential
    def laplacian_mu(xyt):
        # Compute Hessian of mu w.r.t. spatial variables
        hess_mu = jax.hessian(lambda xy_t: mu(jnp.concatenate([xy_t[:2], xyt[2:3]])))(xyt[:2])
        return hess_mu[0, 0] + hess_mu[1, 1]  # ∂²μ/∂x² + ∂²μ/∂y²

    # Vectorize the computations over all coordinates
    dphi_dt = jax.vmap(phi_t)(coords)
    lap_mu = jax.vmap(laplacian_mu)(coords)

    # Cahn-Hilliard equation: ∂φ/∂t = ∇²μ
    residual = dphi_dt - lap_mu

    return residual



def _ch_loss_fn(params, apply_fn, coords_interior, coords_initial, phi_init, coords_boundary_pairs):
    """ Improved loss function with adaptive weighting """
    # PDE residual loss
    residuals = pinn_residual(params, apply_fn, coords_interior)
    loss_pde = jnp.mean(residuals ** 2)

    # Initial condition loss
    pred_init = jax.vmap(lambda x: apply_fn(params, x).squeeze())(coords_initial)
    loss_ic = jnp.mean((pred_init - phi_init.ravel()) ** 2)

    # Conservation of order parameter
    pred_all = jax.vmap(lambda x: apply_fn(params, x).squeeze())(coords_interior)
    mean_phi = jnp.mean(pred_all)
    mean_phi_init = jnp.mean(phi_init)
    loss_conservation = (mean_phi - mean_phi_init) ** 2

    # Periodic boundary conditions
    loss_periodic = 0.0
    for coords_b1, coords_b2 in coords_boundary_pairs:
        pred_b1 = jax.vmap(lambda x: apply_fn(params, x).squeeze())(coords_b1)
        pred_b2 = jax.vmap(lambda x: apply_fn(params, x).squeeze())(coords_b2)
        loss_periodic += jnp.mean((pred_b1 - pred_b2) ** 2)

        # Also match derivatives at boundaries for smoother solutions
        def compute_spatial_grads(coords):
            def phi_at_coords(x):
                return apply_fn(params, x).squeeze()

            def grad_phi(x):
                return jax.grad(phi_at_coords, argnums=0)(x)[:2]  # Just spatial grads

            return jax.vmap(grad_phi)(coords)

        grads_b1 = compute_spatial_grads(coords_b1)
        grads_b2 = compute_spatial_grads(coords_b2)
        loss_periodic += 0.1 * jnp.mean((grads_b1 - grads_b2) ** 2)

    # Adaptive weighting: focus on hardest term
    max_loss = jnp.maximum(loss_pde, jnp.maximum(loss_ic, jnp.maximum(loss_conservation, loss_periodic)))

    # Weight terms by their relative magnitude compared to max loss
    w_pde = 1.0
    w_ic = 10.0 * (max_loss / (loss_ic + 1e-8))
    w_conservation = 1.0 * (max_loss / (loss_conservation + 1e-8))
    w_periodic = 5.0 * (max_loss / (loss_periodic + 1e-8))

    # Cap weights to prevent instability
    max_weight = 100.0
    w_ic = jnp.minimum(w_ic, max_weight)
    w_conservation = jnp.minimum(w_conservation, max_weight)
    w_periodic = jnp.minimum(w_periodic, max_weight)

    total_loss = (w_pde * loss_pde +
                  w_ic * loss_ic +
                  w_conservation * loss_conservation +
                  w_periodic * loss_periodic)
    return total_loss, (loss_pde, loss_ic, loss_conservation, loss_periodic)


@jax.jit
def train_step(state, coords_interior, coords_initial, phi_init, boundary_pairs):
    """Single training step"""

    def _loss_fn(params):
        return _ch_loss_fn(
            params, state.apply_fn, coords_interior, coords_initial,
            phi_init, boundary_pairs
        )

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux_info), grads = grad_fn(state.params)

    # Get gradient norm for monitoring
    grad_norm = optax.global_norm(grads)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    return state, loss, aux_info, grad_norm


def generate_initial_condition(nx=64, ny=64, noise_level=0.1, mean_phi=0.0):
    """
    Generate a physically realistic initial condition for Cahn-Hilliard
    with small random perturbations around the mean value.
    """
    # Create grid
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)

    # Start with small random fluctuations
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, (nx, ny))

    # Create initial condition: mean + noise + some structure
    phi_init = mean_phi + noise_level * noise

    # Add some spatial structure (e.g., a few sinusoidal modes)
    phi_init += 0.2 * jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)
    phi_init += 0.1 * jnp.sin(4 * jnp.pi * X) * jnp.sin(4 * jnp.pi * Y)

    # Ensure the mean is exactly as specified
    current_mean = jnp.mean(phi_init)
    phi_init = phi_init - current_mean + mean_phi

    return phi_init, X, Y


def create_training_points(nx=64, ny=64, nt=20, t_max=1.0, n_interior=10000):
    """
    Create training points for the PINN with proper interior and boundary points.
    """
    # Spatial and temporal grids
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    t = jnp.linspace(0, t_max, nt)

    # Generate random interior points
    key = jax.random.PRNGKey(123)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    x_interior = jax.random.uniform(subkey1, (n_interior,))
    y_interior = jax.random.uniform(subkey2, (n_interior,))
    t_interior = jax.random.uniform(subkey3, (n_interior,)) * t_max

    coords_interior = jnp.stack([x_interior, y_interior, t_interior], axis=1)

    # Initial condition points (t=0)
    X, Y = jnp.meshgrid(x, y)
    x_init, y_init = X.flatten(), Y.flatten()
    t_init = jnp.zeros_like(x_init)
    coords_initial = jnp.stack([x_init, y_init, t_init], axis=1)

    # Boundary points for periodic BCs
    n_boundary = nx * nt  # Number of points along each boundary

    # Create time-space coordinates for each boundary
    t_boundary = jnp.repeat(jnp.linspace(0, t_max, nt), nx)

    # Left boundary (x=0, all y)
    x_left = jnp.zeros(n_boundary)
    y_left = jnp.tile(jnp.linspace(0, 1, nx), nt)
    coords_left = jnp.stack([x_left, y_left, t_boundary], axis=1)

    # Right boundary (x=1, all y)
    x_right = jnp.ones(n_boundary)
    y_right = jnp.tile(jnp.linspace(0, 1, nx), nt)
    coords_right = jnp.stack([x_right, y_right, t_boundary], axis=1)

    # Bottom boundary (all x, y=0)
    x_bottom = jnp.tile(jnp.linspace(0, 1, nx), nt)
    y_bottom = jnp.zeros(n_boundary)
    coords_bottom = jnp.stack([x_bottom, y_bottom, t_boundary], axis=1)

    # Top boundary (all x, y=1)
    x_top = jnp.tile(jnp.linspace(0, 1, nx), nt)
    y_top = jnp.ones(n_boundary)
    coords_top = jnp.stack([x_top, y_top, t_boundary], axis=1)

    # Group boundary pairs for periodic conditions
    boundary_pairs = [
        (coords_left, coords_right),
        (coords_bottom, coords_top)
    ]

    return coords_interior, coords_initial, boundary_pairs

class CahnHilliardPINN:
    def __init__(self, nx=64, ny=64, nt=20, t_max=1.0,
                 network_width=64, network_depth=4,
                 learning_rate=1e-3, kappa=1.0, epsilon=0.9):
        """
        Initialize the Cahn-Hilliard PINN solver

        Args:
            nx, ny: Number of spatial grid points
            nt: Number of time points
            t_max: Maximum simulation time
            network_width: Width of hidden layers
            network_depth: Number of hidden layers
            learning_rate: Initial learning rate
            kappa: Interface parameter in the Cahn-Hilliard equation
            epsilon: Parameter in the Landau free energy
        """
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.t_max = t_max
        self.kappa = kappa
        self.epsilon = epsilon

        # Create network
        features = [network_width] * network_depth + [1]
        self.model = MLPFourier(features=features)  # TODO: If works, pass in B value (also create args class)

        # Initialize training state
        self.train_state = _create_train_state(self.model, learning_rate)

        # Colloction points
        self.coords_interior, self.coords_initial, self.boundary_pairs = create_training_points(
            nx=nx, ny=ny, nt=nt, t_max=t_max, n_interior=10000
        )

        # Generate initial condition
        self.phi_init, self.X, self.Y = generate_initial_condition(nx=nx, ny=ny)

    def train(self, epochs=5000, log_every=100):
        """Train the PINN model"""
        losses = []
        component_losses = []
        grad_norms = []

        start_time = time.time()
        for epoch in range(epochs):
            self.train_state, loss, aux_info, grad_norm = train_step(
                self.train_state,
                self.coords_interior,
                self.coords_initial,
                self.phi_init,
                self.boundary_pairs
            )

            losses.append(loss)
            component_losses.append(aux_info)
            grad_norms.append(grad_norm)

            if epoch % log_every == 0:
                elapsed = time.time() - start_time
                loss_pde, loss_ic, loss_conservation, loss_periodic = aux_info
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4e} - Time: {elapsed:.2f}s")
                print(f"  PDE: {loss_pde:.4e}, IC: {loss_ic:.4e}, "
                      f"Conservation: {loss_conservation:.4e}, Periodic: {loss_periodic:.4e}")
                print(f"  Grad norm: {grad_norm:.4e}")
                start_time = time.time()

                # Early stopping check
                if loss < 1e-6:
                    print("Converged! Stopping early.")
                    break

                # Divergence check
                if jnp.isnan(loss) or loss > 1e20:
                    print("Training diverged. Stopping.")
                    break

        return losses, component_losses, grad_norms

    def predict(self, t_values):
        """
        Predict solution at specified time values

        Args:
            t_values: List of time values to predict at

        Returns:
            List of predicted solutions (phi) at each time
        """
        solutions = []
        x_grid = jnp.linspace(0, 1, self.nx)
        y_grid = jnp.linspace(0, 1, self.ny)
        X, Y = jnp.meshgrid(x_grid, y_grid)
        grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

        for t in t_values:
            # Create space-time points for current time
            t_points = jnp.ones((len(grid_points),)) * t
            st_points = jnp.concatenate([grid_points, t_points[:, None]], axis=1)

            # Predict phi values
            phi_pred = jax.vmap(lambda x: self.model.apply(
                self.train_state.params, x).squeeze())(st_points)

            # Reshape to 2D grid
            phi_grid = phi_pred.reshape(self.ny, self.nx)
            solutions.append(phi_grid)

        return solutions

    def visualize_solution(self, t_values=None):
        """
        Visualize the solution at different time points
        """
        if t_values is None:
            t_values = jnp.linspace(0, self.t_max, 5)

        solutions = self.predict(t_values)

        fig, axes = plt.subplots(1, len(t_values), figsize=(4 * len(t_values), 4))
        if len(t_values) == 1:
            axes = [axes]

        vmin = min(jnp.min(sol) for sol in solutions)
        vmax = max(jnp.max(sol) for sol in solutions)

        im = None
        for i, (t, sol) in enumerate(zip(t_values, solutions)):
            im = axes[i].imshow(sol, origin='lower', extent=[0, 1, 0, 1],
                                vmin=vmin, vmax=vmax, cmap='RdBu')
            axes[i].set_title(f't = {t:.2f}')
            axes[i].set_xlabel('x')
            if i == 0:
                axes[i].set_ylabel('y')

        if im is None:
            print('ERROR')
        else:
            plt.colorbar(im, ax=axes, label='φ')
            plt.tight_layout()
        return fig

    def visualize_loss(self, losses, component_losses):
        """
        Visualize the loss evolution during training
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Total loss
        ax1.semilogy(losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Total Loss Evolution')
        ax1.grid(True)

        # Component losses
        loss_pde = [comp[0] for comp in component_losses]
        loss_ic = [comp[1] for comp in component_losses]
        loss_conservation = [comp[2] for comp in component_losses]
        loss_periodic = [comp[3] for comp in component_losses]

        ax2.semilogy(loss_pde, label='PDE')
        ax2.semilogy(loss_ic, label='Initial')
        ax2.semilogy(loss_conservation, label='Conservation')
        ax2.semilogy(loss_periodic, label='Periodic')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Components')
        ax2.set_title('Loss Components Evolution')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig


class CoupledCahnHilliardPINN:
    """
    A PINN implementation that splits the Cahn-Hilliard equation into two coupled PDEs:
    1. ∂φ/∂t = ∇²μ
    2. μ = ∂f/∂φ - κ∇²φ

    This approach uses two separate networks to model φ and μ.
    """

    def __init__(self, nx=64, ny=64, nt=20, t_max=1.0, network_width=64, network_depth=4):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.t_max = t_max

        # Create networks for both φ and μ
        features = [network_width] * network_depth + [1]
        self.phi_model = MLPFourier(features=features, output_scale=2.0)
        self.mu_model = MLPFourier(features=features, output_scale=5.0)  # μ can have larger range

        # Initialize training states
        self.phi_state = _create_train_state(self.phi_model)
        self.mu_state = _create_train_state(self.mu_model)

        # Generate training points
        self.coords_interior, self.coords_initial, self.boundary_pairs = create_training_points(
            nx=nx, ny=ny, nt=nt, t_max=t_max, n_interior=10000
        )

        # Generate initial condition
        self.phi_init, self.X, self.Y = generate_initial_condition(nx=nx, ny=ny)

        # Parameters
        self.kappa = 1.0
        self.epsilon = 0.9

    def coupled_pinn_residual(self, phi_params, mu_params, coords):
        """
        Calculate residuals for the coupled system.
        """

        def phi_fn(xyt):
            return self.phi_model.apply(phi_params, xyt).squeeze()

        def mu_fn(xyt):
            return self.mu_model.apply(mu_params, xyt).squeeze()

        # First equation: ∂φ/∂t = ∇²μ
        def dphi_dt(xyt):
            return jax.grad(phi_fn, argnums=0)(xyt)[2]  # w.r.t time coordinate

        def laplacian_mu(xyt):
            hess_mu = jax.hessian(lambda xy: mu_fn(jnp.concatenate([xy, xyt[2:3]])))(xyt[:2])
            return hess_mu[0, 0] + hess_mu[1, 1]

        # Second equation: μ = ∂f/∂φ - κ∇²φ
        def chemical_potential(xyt):
            phi = phi_fn(xyt)
            hess_phi = jax.hessian(lambda xy: phi_fn(jnp.concatenate([xy, xyt[2:3]])))(xyt[:2])
            lap_phi = hess_phi[0, 0] + hess_phi[1, 1]
            return dfdphi(phi, self.epsilon) - self.kappa * lap_phi

        # Vectorize computations
        dphi_dt_values = jax.vmap(dphi_dt)(coords)
        lap_mu_values = jax.vmap(laplacian_mu)(coords)
        mu_values = jax.vmap(mu_fn)(coords)
        chemical_potential_values = jax.vmap(chemical_potential)(coords)

        # Residuals for both equations
        residual_eq1 = dphi_dt_values - lap_mu_values  # ∂φ/∂t = ∇²μ
        residual_eq2 = mu_values - chemical_potential_values  # μ = ∂f/∂φ - κ∇²φ

        return residual_eq1, residual_eq2

    def coupled_loss_fn(self, phi_params, mu_params):
        """
        Compute loss for the coupled PINN approach.
        """
        # PDE residuals
        residual_eq1, residual_eq2 = self.coupled_pinn_residual(
            phi_params, mu_params, self.coords_interior
        )
        loss_pde1 = jnp.mean(residual_eq1 ** 2)
        loss_pde2 = jnp.mean(residual_eq2 ** 2)

        # Initial condition loss for φ
        phi_init_pred = jax.vmap(lambda x: self.phi_model.apply(phi_params, x).squeeze())(self.coords_initial)
        loss_ic = jnp.mean((phi_init_pred - self.phi_init.ravel()) ** 2)

        # Consistency loss for μ at t=0
        mu_init_pred = jax.vmap(lambda x: self.mu_model.apply(mu_params, x).squeeze())(self.coords_initial)

        # Compute expected μ at t=0 from the initial φ
        def expected_mu_at_init(x):
            phi = self.phi_init.ravel()[x]
            # Approximate Laplacian using finite difference (simplified)
            laplacian_phi = 0.0  # This is a simplification
            return dfdphi(phi, self.epsilon) - self.kappa * laplacian_phi

        expected_mu = jnp.vectorize(expected_mu_at_init)(jnp.arange(len(self.phi_init.ravel())))
        loss_mu_ic = jnp.mean((mu_init_pred - expected_mu) ** 2)

        # Conservation law
        phi_pred = jax.vmap(lambda x: self.phi_model.apply(phi_params, x).squeeze())(self.coords_interior)
        mean_phi = jnp.mean(phi_pred)
        mean_phi_init = jnp.mean(self.phi_init)
        loss_conservation = (mean_phi - mean_phi_init) ** 2

        # Periodic boundary conditions for both φ and μ
        loss_periodic = 0.0
        for coords_b1, coords_b2 in self.boundary_pairs:
            # Periodic BCs for φ
            phi_b1 = jax.vmap(lambda x: self.phi_model.apply(phi_params, x).squeeze())(coords_b1)
            phi_b2 = jax.vmap(lambda x: self.phi_model.apply(phi_params, x).squeeze())(coords_b2)
            loss_periodic += jnp.mean((phi_b1 - phi_b2) ** 2)

            # Periodic BCs for μ
            mu_b1 = jax.vmap(lambda x: self.mu_model.apply(mu_params, x).squeeze())(coords_b1)
            mu_b2 = jax.vmap(lambda x: self.mu_model.apply(mu_params, x).squeeze())(coords_b2)
            loss_periodic += jnp.mean((mu_b1 - mu_b2) ** 2)

        # Total loss with weighted components
        total_loss = (
                1.0 * loss_pde1 +
                1.0 * loss_pde2 +
                10.0 * loss_ic +
                1.0 * loss_mu_ic +
                1.0 * loss_conservation +
                5.0 * loss_periodic
        )

        return total_loss, (loss_pde1, loss_pde2, loss_ic, loss_mu_ic, loss_conservation, loss_periodic)

    @jax.jit
    def train_step(self, phi_state, mu_state):
        """Single training step for both networks"""

        def loss_fn(phi_params, mu_params):
            return self.coupled_loss_fn(phi_params, mu_params)

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
        (loss, aux_info), (phi_grads, mu_grads) = grad_fn(phi_state.params, mu_state.params)

        # Update parameters
        phi_state = phi_state.apply_gradients(grads=phi_grads)
        mu_state = mu_state.apply_gradients(grads=mu_grads)

        return phi_state, mu_state, loss, aux_info

    def train(self, epochs=5000, log_every=100):
        """Train both networks simultaneously"""
        losses = []
        component_losses = []

        for epoch in range(epochs):
            self.phi_state, self.mu_state, loss, aux_info = self.train_step(
                self.phi_state, self.mu_state
            )

            losses.append(loss)
            component_losses.append(aux_info)

            if epoch % log_every == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4e}")
                print(f"  PDE1: {aux_info[0]:.4e}, PDE2: {aux_info[1]:.4e}, "
                      f"IC: {aux_info[2]:.4e}, μIC: {aux_info[3]:.4e}, "
                      f"Conservation: {aux_info[4]:.4e}, Periodic: {aux_info[5]:.4e}")

        return losses, component_losses

    def predict(self, t_values):
        """Predict both φ and μ at specified time values"""
        phi_solutions = []
        mu_solutions = []

        x_grid = jnp.linspace(0, 1, self.nx)
        y_grid = jnp.linspace(0, 1, self.ny)
        X, Y = jnp.meshgrid(x_grid, y_grid)
        grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

        for t in t_values:
            # Create space-time points
            t_points = jnp.ones((len(grid_points),)) * t
            st_points = jnp.concatenate([grid_points, t_points[:, None]], axis=1)

            # Predict φ values
            phi_pred = jax.vmap(lambda x: self.phi_model.apply(
                self.phi_state.params, x).squeeze())(st_points)
            phi_grid = phi_pred.reshape(self.ny, self.nx)
            phi_solutions.append(phi_grid)

            # Predict μ values
            mu_pred = jax.vmap(lambda x: self.mu_model.apply(
                self.mu_state.params, x).squeeze())(st_points)
            mu_grid = mu_pred.reshape(self.ny, self.nx)
            mu_solutions.append(mu_grid)

        return phi_solutions, mu_solutions