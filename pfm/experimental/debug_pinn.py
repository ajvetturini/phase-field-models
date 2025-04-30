import jax
import jax.numpy as jnp

def debug_gradient_issues(model, coords, dfdphi, params=None):
    """
    Debug potential gradient instability issues by examining gradient magnitudes
    at different parts of the computation.
    """
    if params is None:
        # Initialize parameters if not provided
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))

    # Test function to get phi values
    def phi_fn(xyt):
        return model.apply(params, xyt).squeeze()

    # 1. Check phi values
    phi_values = jax.vmap(phi_fn)(coords)
    print(f"phi values - min: {jnp.min(phi_values)}, max: {jnp.max(phi_values)}, mean: {jnp.mean(phi_values)}")

    # 2. Check first derivatives
    def get_first_derivatives(xyt):
        return jax.grad(phi_fn)(xyt)

    grads = jax.vmap(get_first_derivatives)(coords)
    print(f"First derivatives - min: {jnp.min(grads)}, max: {jnp.max(grads)}, mean: {jnp.mean(jnp.abs(grads))}")

    # 3. Check second derivatives (for Laplacian)
    def get_laplacian(xyt):
        hess = jax.hessian(phi_fn)(xyt)
        return hess[0, 0] + hess[1, 1]  # ∂²φ/∂x² + ∂²φ/∂y²

    laplacians = jax.vmap(get_laplacian)(coords)
    print(f"Laplacians - min: {jnp.min(laplacians)}, max: {jnp.max(laplacians)}, mean: {jnp.mean(jnp.abs(laplacians))}")

    # 4. Check chemical potential
    def get_chemical_potential(xyt):
        phi = phi_fn(xyt)
        lap = get_laplacian(xyt)
        kappa = 1.0
        return dfdphi(phi) - kappa * lap

    mu_values = jax.vmap(get_chemical_potential)(coords)
    print(
        f"Chemical potential - min: {jnp.min(mu_values)}, max: {jnp.max(mu_values)}, mean: {jnp.mean(jnp.abs(mu_values))}")

    # 5. Check laplacian of chemical potential (the most complex part)
    def get_laplacian_mu(xyt):
        hess_mu = jax.hessian(lambda xy_t: get_chemical_potential(jnp.concatenate([xy_t[:2], xyt[2:3]])))(xyt[:2])
        return hess_mu[0, 0] + hess_mu[1, 1]  # ∂²μ/∂x² + ∂²μ/∂y²

    try:
        lap_mu_values = jax.vmap(get_laplacian_mu)(coords[:10])  # Use fewer points for this complex operation
        print(
            f"Laplacian of mu - min: {jnp.min(lap_mu_values)}, max: {jnp.max(lap_mu_values)}, mean: {jnp.mean(jnp.abs(lap_mu_values))}")
    except Exception as e:
        print(f"Error computing Laplacian of mu: {e}")

    return {
        "phi": phi_values,
        "grads": grads,
        "laplacians": laplacians,
        "mu": mu_values
    }


def verify_conservation_law(model, params, nx=32, ny=32, t_values=None):
    """
    Verify that the PINN solution conserves the order parameter globally.
    """
    if t_values is None:
        t_values = jnp.linspace(0, 1.0, 10)

    # Create spatial grid
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    # Function to compute mean phi at a given time
    def compute_mean_phi(t):
        t_points = jnp.ones((len(grid_points),)) * t
        st_points = jnp.concatenate([grid_points, t_points[:, None]], axis=1)
        phi_values = jax.vmap(lambda x: model.apply(params, x).squeeze())(st_points)
        return jnp.mean(phi_values)

    # Compute mean phi for each time
    mean_phi_values = jnp.array([compute_mean_phi(t) for t in t_values])

    # Calculate variation from initial value
    variation = mean_phi_values - mean_phi_values[0]
    max_variation = jnp.max(jnp.abs(variation))

    print(f"Conservation check - Max variation in mean φ: {max_variation:.6e}")
    print(f"Mean φ values: {mean_phi_values}")

    return mean_phi_values, variation