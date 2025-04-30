from pfm.pinn import CahnHilliardPINN
import jax
import jax.numpy as jnp
import optax


def solve_with_incremental_complexity(final_epochs=5000):
    """
    Train the PINN using an incremental complexity approach.
    Start with a simplified problem and gradually increase complexity.

    1. First train with only initial conditions (static problem)
    2. Then add time evolution with small time window
    3. Finally train the full problem
    """
    # Step 1: Configure a static problem (no time evolution)
    print("=== Stage 1: Training on static problem (initial condition only) ===")
    static_pinn = CahnHilliardPINN(nx=32, ny=32, nt=2, t_max=0.001)

    # Modify the loss function to focus only on initial condition
    @jax.jit
    def static_train_step(state, coords_initial, phi_init, boundary_pairs):
        def loss_fn(params):
            # Only compute initial condition loss
            pred_init = jax.vmap(lambda x: state.apply_fn(params, x).squeeze())(coords_initial)
            loss_ic = jnp.mean((pred_init - phi_init.ravel()) ** 2)

            # Add minimal periodic boundary condition loss
            loss_periodic = 0.0
            for coords_b1, coords_b2 in boundary_pairs:
                pred_b1 = jax.vmap(lambda x: state.apply_fn(params, x).squeeze())(coords_b1)
                pred_b2 = jax.vmap(lambda x: state.apply_fn(params, x).squeeze())(coords_b2)
                loss_periodic += jnp.mean((pred_b1 - pred_b2) ** 2)

            return loss_ic + 0.1 * loss_periodic, (loss_ic, jnp.array(0.0), jnp.array(0.0), loss_periodic)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux_info), grads = grad_fn(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux_info, grad_norm

    # Train static problem
    for epoch in range(500):
        static_pinn.train_state, loss, aux_info, grad_norm = static_train_step(
            static_pinn.train_state,
            static_pinn.coords_initial,
            static_pinn.phi_init,
            static_pinn.boundary_pairs
        )

        if epoch % 100 == 0:
            print(f"Stage 1 - Epoch {epoch}/500 - Loss: {loss:.4e}")

    # Step 2: Train with small time window
    print("\n=== Stage 2: Training with small time window ===")
    small_t_pinn = CahnHilliardPINN(nx=32, ny=32, nt=10, t_max=0.1)

    # Initialize with parameters from static training
    small_t_pinn.train_state = small_t_pinn.train_state.replace(params=static_pinn.train_state.params)

    # Train with small time window
    small_t_losses, _, _ = small_t_pinn.train(epochs=1000, log_every=200)

    # Step 3: Train full problem with initialized parameters
    print("\n=== Stage 3: Training full problem ===")
    full_pinn = CahnHilliardPINN(nx=64, ny=64, nt=20, t_max=1.0)

    # Initialize with parameters from small time window training
    full_pinn.train_state = full_pinn.train_state.replace(params=small_t_pinn.train_state.params)

    # Train full problem
    full_losses, component_losses, grad_norms = full_pinn.train(epochs=final_epochs, log_every=100)

    return full_pinn, full_losses, component_losses


def explore_fourier_feature_scaling(scales=[1.0, 2.0, 5.0, 10.0], epochs=500):
    """
    Test different Fourier feature scaling parameters to find optimal settings.
    Fourier feature scaling affects how the network learns different frequency components.
    """
    from pfm.pinn.networks import MLPFourier
    from pfm.pinn.ch_pinn import _create_train_state
    results = {}

    for scale in scales:
        print(f"\n=== Testing Fourier scale: {scale} ===")

        # Create model with specific Fourier scale
        features = [64, 64, 64, 64, 1]  # Example network architecture
        model = MLPFourier(features=features, B_scale=scale)

        # Setup minimal training
        pinn = CahnHilliardPINN(nx=32, ny=32, nt=10, t_max=0.5)
        pinn.model = model
        pinn.train_state = _create_train_state(model)

        # Train for fixed number of epochs
        losses, _, _ = pinn.train(epochs=epochs, log_every=100)

        # Store results
        results[scale] = {
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'losses': losses
        }

        print(f"Scale {scale} - Final loss: {losses[-1]:.4e}, Min loss: {min(losses):.4e}")

    # Find best scale
    best_scale = min(scales, key=lambda s: results[s]['min_loss'])
    print(f"\nBest Fourier scale: {best_scale} with min loss: {results[best_scale]['min_loss']:.4e}")

    return results, best_scale

epochs = 10
nx = 64
ny = 64
nt = 20
t_max = 1.0

# Create solver
ch_pinn = CahnHilliardPINN(nx=nx, ny=ny, nt=nt, t_max=t_max)

# Train model
losses, component_losses, grad_norms = ch_pinn.train(epochs=epochs)

# Visualize results
fig_loss = ch_pinn.visualize_loss(losses, component_losses)
fig_solution = ch_pinn.visualize_solution()

