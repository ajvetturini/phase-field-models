""" Main function call to train the passed in PINN network """
import jax
import optax
import jax.numpy as jnp

def _cahn_hilliard_residual(model, params, xyt, free_energy_model, kappa, n_species):
    """ Residuals cacluation for training Cahn-Hilliard network """
    def rho_and_mu_fn(xyt_in):
        # Model outputs [rho_1, ..., rho_N, mu_1, ..., mu_N]
        model_output = model.apply(params, xyt_in)
        rho = model_output[:n_species]
        mu = model_output[n_species:]
        return rho, mu

    def rho_fn(xyt_in):
        return rho_and_mu_fn(xyt_in)[0]

    def mu_fn(xyt_in):
        return rho_and_mu_fn(xyt_in)[1]

    def single_residual(xyt_in):
        # rho and its derivatives
        rho_val = rho_fn(xyt_in)  # Value of rho
        J_rho = jax.jacfwd(rho_fn)(xyt_in)  # Jacobian of rho wrt (x, y, t)
        rho_t = J_rho[..., -1]  # Time derivative: ∂ρ/∂t
        hessian_rho = jax.hessian(rho_fn)(xyt_in)

        # Slicing [..., :-1, :-1] removes the time dimension from the Hessian
        lap_rho = jnp.trace(hessian_rho[..., :-1, :-1], axis1=-2, axis2=-1)

        # mu and its derivatives
        mu_val = mu_fn(xyt_in)  # Value of mu
        hessian_mu = jax.hessian(mu_fn)(xyt_in)
        lap_mu = jnp.trace(hessian_mu[..., :-1, :-1], axis1=-2, axis2=-1)

        # Residual 1: R₁ = ∂ρ/∂t - ∇²μ
        residual_1 = rho_t - lap_mu

        # Residual 2: R₂ = μ - (df/dρ - κ ∇²ρ)
        bulk_derivative = free_energy_model.der_bulk_free_energy(rho_val)
        residual_2 = mu_val - (bulk_derivative - kappa * lap_rho)

        # Concatenate residuals for all species
        return jnp.concatenate([residual_1, residual_2], axis=-1)

    # Vmap over the entire batch of collocation points
    residuals = jax.vmap(single_residual)(xyt)
    return residuals

def train_ch(config, model, free_energy_model, total_system, N_species):
    """ Train Cahn_Hilliard-based PINN """
    train_params = _read_in_config(config)
    rng = jax.random.PRNGKey(train_params.get('seed'))
    kappa = total_system.k_laplacian

    # Initialize parameters with dummy input
    xyt_dummy = jnp.ones((1, total_system.dim + 1))
    params = model.init(rng, xyt_dummy)

    # Setup optimizer + model:
    optimizer = optax.adam(train_params.get('learning_rate'))
    opt_state = optimizer.init(params)

    # Collocation points (sampled over domain + time)
    Ncolloc = train_params.get('n_collocation')
    xyt = jax.random.uniform(rng, (Ncolloc, total_system.dim + 1))

    def loss_fn(_params, _xyt):
        R = _cahn_hilliard_residual(model, _params, _xyt, free_energy_model, kappa, N_species)
        return jnp.mean(R ** 2)

    @jax.jit
    def train_step(_params, _opt_state, _xyt):
        cur_loss, grads = jax.value_and_grad(loss_fn)(_params, _xyt)
        updates, next_opt_state = optimizer.update(grads, _opt_state, _params)
        next_params = optax.apply_updates(_params, updates)
        return next_params, next_opt_state, cur_loss


    for epoch in range(train_params.get('epochs')):
        params, opt_state, loss = train_step(params, opt_state, xyt)
        if epoch % 100 == 0:
            print(f"Step {epoch}, Loss = {loss:.6f}")

    return params


def _read_in_config(config):
    """ Get training parameters from config """
    train_config = config.get('pinn_training_parameters', {})
    train_params = {
        'epochs': train_config.get('epochs', 1000),
        'batch_size': train_config.get('batch_size', 32),
        'learning_rate': train_config.get('learning_rate', 0.001),
        'weight_decay': train_config.get('weight_decay', 0.0),
        'log_frequency': train_config.get('log_frequency', 100),
        'seed': train_config.get('seed', 8),
        'n_collocation': train_config.get('n_collocation', 512)
    }
    return train_params

def train_ac(config, network, free_energy_model, total_system, N_species):
    """ Train Allen_Cahn-based PINN """
    train_params = _read_in_config(config)
    raise NotImplementedError('WIP')