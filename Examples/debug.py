import jax
import jax.numpy as jnp

def elementwise_bulk_free_energy(rho):
    return -0.5 * 1.0 * rho ** 2 + 0.25 * rho ** 4

# Define a function to compute gradient
grad_energy = jax.grad(elementwise_bulk_free_energy)

# Create your 2D field
rho_field_2d = jnp.ones((64, 64))

# Apply vectorized gradient calculation over the entire field
vectorized_grad = jax.vmap(jax.vmap(grad_energy))
gradient_field = vectorized_grad(rho_field_2d)

# Alternatively, you can compute a single gradient field directly:
def total_energy(field):
    return jnp.sum(elementwise_bulk_free_energy(field))

field_gradient = jax.grad(total_energy)(rho_field_2d)