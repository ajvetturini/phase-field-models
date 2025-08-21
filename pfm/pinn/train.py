""" Main function call to train the passed in PINN network """
import jax
import optax
import jax.numpy as jnp

def _cahn_hilliard_residual(model, params, xyt, free_energy_model, interface_scalar, kappa, mobility, n_species):
    """ Residuals cacluation for training Cahn-Hilliard network """
    def rho_mu_fn_single(xyt_point):
        # Model outputs [rho_1, ..., rho_N, mu_1, ..., mu_N]
        model_output = model.apply(params, xyt_point)   # shape: (2*n_species,)
        rho, mu = jnp.split(model_output, 2)
        return rho, mu

    def compute_rho_laplacians(xyt_point):
        # Helper function: computes Laplacian for the i-th rho at a point z.
        def _lap(i, z):
            hessian_matrix = jax.hessian(lambda pt: rho_mu_fn_single(pt)[0][i])(z)
            return hessian_matrix[0, 0] + hessian_matrix[1, 1]
        return jax.vmap(_lap, in_axes=(0, None))(jnp.arange(n_species), xyt_point)

    def compute_mu_laplacians(xyt_point):
        # Same logic for mu.
        def _lap(i, z):
            hessian_matrix = jax.hessian(lambda pt: rho_mu_fn_single(pt)[1][i])(z)
            return hessian_matrix[0, 0] + hessian_matrix[1, 1]
        return jax.vmap(_lap, in_axes=(0, None))(jnp.arange(n_species), xyt_point)


    def single_residual(xyt_point):
        # Get values and all derivatives at the point
        rho_val, mu_val = rho_mu_fn_single(xyt_point)  # Each has shape (n_species,)
        jac_rho, jac_mu = jax.jacobian(rho_mu_fn_single)(xyt_point)  # Shape (n_species, 3) for each
        lap_rho_val = compute_rho_laplacians(xyt_point)
        lap_mu_val = compute_mu_laplacians(xyt_point)

        # Extract time derivative from the Jacobian
        rho_t = jac_rho[:, -1]

        # Residual 1: ∂ρ/∂t - ∇²μ = 0  (assuming mobility M=1)
        residual_1 = rho_t - mobility * lap_mu_val

        # Residual 2: μ - (∂f_bulk/∂ρ - κ∇²ρ) = 0
        bulk_derivative = free_energy_model.der_bulk_free_energy_pointwise(rho_val)
        residual_2 = mu_val - (bulk_derivative - interface_scalar * kappa * lap_rho_val)

        return jnp.concatenate([residual_1, residual_2], axis=-1)

    # Vmap the entire single-point calculation over the batch of collocation points.
    # This is the key to efficiency.
    residuals = jax.vmap(single_residual)(xyt)
    return residuals

def train_ch(config, model, free_energy_model, total_system, N_species, initial_condition):
    """ Train Cahn_Hilliard-based PINN """
    train_params = _read_in_config(config)
    rng = jax.random.PRNGKey(train_params.get('seed'))
    kappa = total_system.k_laplacian
    mobility = total_system.M
    w_ic = train_params.get('w_ic', 100.0)
    w_bc = train_params.get('w_bc', 10.0)
    kappa_weight = config.get('interface_scalar', 1.0)

    # Initialize parameters with dummy input
    xyt_dummy = jnp.ones((1, total_system.dim + 1))
    params = model.init(rng, xyt_dummy)

    # Setup optimizer + model:
    lr_schedule = optax.exponential_decay(
        init_value=train_params.get('learning_rate'),
        transition_steps=1000,
        decay_rate=0.95,
        end_value=1e-6
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=train_params.get('weight_decay'))
    opt_state = optimizer.init(params)

    # Setup positional bounds:
    t_bounds = [0.0, 1.0]  # Normalized time & we want the steady state (i.e., t=1.0)
    x_bounds = [0.0, 1.0]
    y_bounds = [0.0, 1.0]

    # Use initial condition to get IC for PINN training
    # Initial condition is of shape (N_species, Nx, Ny)
    x_space = jnp.linspace(x_bounds[0], x_bounds[1], initial_condition.shape[1])
    y_space = jnp.linspace(y_bounds[0], y_bounds[1], initial_condition.shape[2])
    xx, yy = jnp.meshgrid(x_space, y_space)
    xy_initial = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)
    rho_initial_flat = jnp.transpose(initial_condition, (1, 2, 0)).reshape(-1, N_species)  # N_bins, N_species

    def loss_fn(_params, _pde_pts, _ic_pts, _ic_rho, _bc_pts_pair):
        # First get residual loss:
        R = _cahn_hilliard_residual(model, _params, _pde_pts, free_energy_model, kappa_weight, kappa, mobility,
                                    N_species)
        mean_residual_squared = jnp.mean(R ** 2)

        # Initial condition loss
        t0_pts = jnp.concatenate([_ic_pts, jnp.zeros((_ic_pts.shape[0], 1))], axis=-1)
        pred_rho_t0 = model.apply(_params, t0_pts)[:, :N_species]  # Only need rho
        loss_ic = jnp.mean((pred_rho_t0 - _ic_rho) ** 2)

        # Periodic boundary condition Loss
        # Enforces that rho(x_min, y, t) == rho(x_max, y, t) and
        # rho(x, y_min, t) == rho(x, y_max, t)
        bc_pts_1, bc_pts_2, bc_pts_3, bc_pts_4 = _bc_pts_pair

        def gradient_fn(pts):
            return jax.jacrev(lambda p: model.apply(_params, p)[:, :N_species])(pts)

        # First ensure field parameter is periodic
        pred_1 = model.apply(_params, bc_pts_1)[:, :N_species]
        pred_2 = model.apply(_params, bc_pts_2)[:, :N_species]
        pred_3 = model.apply(_params, bc_pts_3)[:, :N_species]
        pred_4 = model.apply(_params, bc_pts_4)[:, :N_species]
        loss_bc = jnp.mean((pred_1 - pred_2) ** 2) + jnp.mean((pred_3 - pred_4) ** 2)

        # Then ensure gradient of field parameter is also periodic
        grad_1 = gradient_fn(bc_pts_1)
        grad_2 = gradient_fn(bc_pts_2)
        grad_3 = gradient_fn(bc_pts_3)
        grad_4 = gradient_fn(bc_pts_4)
        loss_bc += jnp.mean((grad_1 - grad_2) ** 2) + jnp.mean((grad_3 - grad_4) ** 2)

        weighted_ic_loss = w_ic * loss_ic
        weighted_bc_loss = w_bc * loss_bc
        total_loss = mean_residual_squared + weighted_ic_loss + weighted_bc_loss
        loss_dict = {
            'total': total_loss,
            'pde': mean_residual_squared,
            'ic': weighted_ic_loss,
            'bc': weighted_bc_loss
        }
        return total_loss, loss_dict

    @jax.jit
    def train_step(_params, _opt_state, _pde_pts, _ic_pts, _ic_rho, _bc_pts_pair):
        (cur_loss, loss_breakdown), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            _params, _pde_pts, _ic_pts, _ic_rho, _bc_pts_pair
        )
        updates, next_opt_state = optimizer.update(grads, _opt_state, _params)
        next_params = optax.apply_updates(_params, updates)
        return next_params, next_opt_state, loss_breakdown

    n_collocation = train_params.get('n_collocation')
    n_bc = train_params.get('n_boundary')

    for epoch in range(train_params.get('epochs')):
        rng, pde_key, bcx_key, bcy_key, bct_key = jax.random.split(rng, 5)  # Resample points at each epoch for better training

        # Collocation points for PDE:
        pde_pts = jax.random.uniform(pde_key, (n_collocation, total_system.dim + 1))
        pde_pts = pde_pts * jnp.array([
            x_bounds[1] - x_bounds[0],
            y_bounds[1] - y_bounds[0],
            t_bounds[1] - t_bounds[0]
        ]) + jnp.array([x_bounds[0], y_bounds[0], t_bounds[0]])

        # Sample one set of boundary bounds / points:
        t_bc = jax.random.uniform(bct_key, (n_bc, 1)) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        y_bc = jax.random.uniform(bcy_key, (n_bc, 1)) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
        x_bc = jax.random.uniform(bcx_key, (n_bc, 1)) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]

        # Build points arrays for periodic BCs
        bc_x_1 = jnp.concatenate([jnp.full_like(y_bc, x_bounds[0]), y_bc, t_bc], axis=1)
        bc_x_2 = jnp.concatenate([jnp.full_like(y_bc, x_bounds[1]), y_bc, t_bc], axis=1)
        bc_y_1 = jnp.concatenate([x_bc, jnp.full_like(x_bc, y_bounds[0]), t_bc], axis=1)
        bc_y_2 = jnp.concatenate([x_bc, jnp.full_like(x_bc, y_bounds[1]), t_bc], axis=1)

        bc_pts_pair = (bc_x_1, bc_x_2, bc_y_1, bc_y_2)

        params, opt_state, losses = train_step(
            params, opt_state, pde_pts, xy_initial, rho_initial_flat, bc_pts_pair
        )
        if epoch % 100 == 0:
            print(f"Step {epoch} | Total Loss = {losses['total']:.4f} | PDE = {losses['pde']:.4f} | "
                  f"IC = {losses['ic']:.4f} | BC = {losses['bc']:.4f}")

    return params


def _read_in_config(config):
    """ Get training parameters from config """
    train_config = config.get('pinn_training_parameters', {})
    train_params = {
        'epochs': train_config.get('epochs', 10000),
        # 'batch_size': train_config.get('batch_size', 32),
        'learning_rate': train_config.get('learning_rate', 0.001),
        'weight_decay': train_config.get('weight_decay', 1e-5),
        'log_frequency': train_config.get('log_frequency', 100),
        'seed': train_config.get('seed', 8),
        'n_collocation': train_config.get('n_collocation', 8192),
        'n_boundary': train_config.get('n_boundary', 1024),
        'w_ic': train_config.get('w_ic', 100.0),  # Initial condition weighting for composite loss function
        'w_bc': train_config.get('w_bc', 10.0),   # Boundary condition weight for composite loss function
    }
    return train_params

def train_ac(config, network, free_energy_model, total_system, N_species):
    """ Train Allen_Cahn-based PINN """
    train_params = _read_in_config(config)
    raise NotImplementedError('WIP')