"""
Perform Cahn-Hilliard phase field modeling using Numba and NumPy.

Overall, a Wertheim theory based energy functional is used to model the phase separation of a binary fluid mixture.
The Cahn-Hilliard equation is solved using a 1) an explicit euler and 2) a semi-implicit Fourier spectral method.
The code is designed to be efficient and easy to use, with a focus on performance and readability.
"""
import numpy as np
from numba import cuda


def laplacian(psi, dx):
    """ 2D Laplacian operator using finite differences. """
    # NOTE: Psi is a 3D array with shape (N_species, N, N)
    xm = np.roll(psi, shift=1, axis=1)
    xp = np.roll(psi, shift=-1, axis=1)
    ym = np.roll(psi, shift=1, axis=2)
    yp = np.roll(psi, shift=-1, axis=2)

    lap = (xm + xp + ym + yp - 4.0 * psi) / (dx ** 2)

    return lap


def _X(rho):
    """ Calculates fraction of molecules that are bonded (or unbounded) using the valence delta specified """
    dtype_eps = np.array(1e-6, dtype=rho.dtype)
    two_valence_delta = 173174.9910

    arg = 2.0 * two_valence_delta * rho
    use_taylor = arg < np.array(1e-5, dtype=rho.dtype)
    X_stable = 2.0 / np.maximum(1.0 + np.sqrt(np.maximum(1.0 + arg, 0.0)), dtype_eps)
    X_taylor = 0.5 - 0.25 * two_valence_delta * rho + 0.125 * (two_valence_delta * rho) ** 2  # Higher order if needed
    X_val = np.where(use_taylor, X_taylor, X_stable)  # Choose best based on magnitude

    return np.maximum(X_val, 0.0).astype(rho.dtype)


def der_bulk_energy(phi):
    """ Calculate the derivative of the bulk energy functional. """
    # NOTE: phi is a 3D array with shape (N_species, Nx, Ny)
    # NOTE: There are hard-coded values to this specific model, this is a debug script.

    # rho_all_species has shape (1, Nx, Ny)
    threshold = np.array(1e-6, dtype=phi.dtype)  # Slightly larger than 0 threshold to use for stability
    der_f_ref = np.where(
        phi < threshold,
        phi / threshold + np.array(-21.7232, dtype=phi.dtype),
        np.log(np.maximum(phi, threshold))  # Stability / safety so no nan
    )
    der_f_ref += (2 * 2190 * phi)

    X = _X(phi)  # Ensure _X is vectorized
    log_X_arg = np.log(np.maximum(X, threshold))

    der_f_bond = np.where(
        phi > threshold,
        4 * log_X_arg,
        0.0,
    )

    return der_f_bond + der_f_ref


def explicit_euler(init_phi, nsteps, dt, dx):
    """ Perform explicit Euler time-stepping for the Cahn-Hilliard equation. """
    phi = init_phi.copy()
    for _ in range(nsteps):
        lap_phi = laplacian(phi, dx)
        bulk_energy = der_bulk_energy(phi)
        chemical_potential = bulk_energy - (2.0 * 1e6 * lap_phi)  # Specific to Wertheim formulation
        lap_d_phi = laplacian(chemical_potential, dx)
        phi += dt * lap_d_phi

    return phi

@cuda.jit
def laplacian_gpu(psi, dx, lap):
    """ 2D Laplacian operator using finite differences - GPU version. """
    n_species, ny, nx = psi.shape
    i, j, s = cuda.grid(3)  # 3D grid for species and spatial dimensions

    if 0 < i < ny - 1 and 0 < j < nx - 1 and 0 <= s < n_species:
        xm = psi[s, i - 1, j]
        xp = psi[s, i + 1, j]
        ym = psi[s, i, j - 1]
        yp = psi[s, i, j + 1]
        lap[s, i, j] = (xm + xp + ym + yp - 4.0 * psi[s, i, j]) / (dx ** 2)

@cuda.jit
def _X_gpu(rho, X_val):
    """ Calculates fraction of molecules that are bonded (or unbounded) using the valence delta specified """
    i, j, s = cuda.grid(3)
    if 0 <= i < rho.shape[1] and 0 <= j < rho.shape[2] and 0 <= s < rho.shape[0]:
        dtype_eps = 1e-6
        two_valence_delta = 173174.9910

        arg = 2.0 * two_valence_delta * rho[s, i, j]
        use_taylor = arg < 1e-5
        X_stable = 2.0 / max(1.0 + (max(1.0 + arg, 0.0)**0.5), dtype_eps)
        X_taylor = 0.5 - 0.25 * two_valence_delta * rho[s, i, j] + 0.125 * (two_valence_delta * rho[s, i, j]) ** 2
        X_val[s, i, j] = max(X_taylor if use_taylor else X_stable, 0.0)

@cuda.jit
def der_bulk_energy_gpu(phi, der_f):
    """ Calculate the derivative of the bulk energy functional - GPU version. """
    i, j, s = cuda.grid(3)
    if 0 <= i < phi.shape[1] and 0 <= j < phi.shape[2] and 0 <= s < phi.shape[0]:
        threshold = 1e-6  # Slightly larger than 0 threshold to use for stability
        if phi[s, i, j] < threshold:
            der_f_ref = phi[s, i, j] / threshold + (-21.7232)
        else:
            der_f_ref = np.log(max(phi[s, i, j], threshold))  # Stability / safety so no nan
        der_f_ref += (2 * 2190 * phi[s, i, j])

        X_val_gpu = cuda.device_array_like(phi)
        _X_gpu[cuda.grid.shape(3), cuda.grid.thread_count(3)](phi, X_val_gpu)
        log_X_arg = cuda.local.array(1, dtype=phi.dtype)  # Use local memory
        if X_val_gpu[s, i, j] > threshold:
            log_X_arg[0] = np.log(X_val_gpu[s, i, j])
            der_f_bond = 4 * log_X_arg[0]
        else:
            der_f_bond = 0.0

        der_f[s, i, j] = der_f_bond + der_f_ref

@cuda.jit(device=True)
def _elementwise_subtract_scaled(a, b, scalar, out):
    i, j, s = cuda.grid(3)
    if 0 <= s < a.shape[0] and 0 <= i < a.shape[1] and 0 <= j < a.shape[2]:
        out[s, i, j] = a[s, i, j] + scalar * b[s, i, j]

@cuda.jit(device=True)
def _elementwise_add_scaled(a, b, scalar, out):
    i, j, s = cuda.grid(3)
    if 0 <= s < a.shape[0] and 0 <= i < a.shape[1] and 0 <= j < a.shape[2]:
        out[s, i, j] = a[s, i, j] + scalar * b[s, i, j]

def explicit_euler_gpu(phi_gpu, nsteps, dt, dx):
    """ Perform explicit Euler time-stepping for the Cahn-Hilliard equation - GPU version. """
    n_species, ny, nx = phi_gpu.shape

    threadsperblock = (16, 16, 1)
    blockspergrid_y = (ny + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_x = (nx + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_s = (n_species + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_y, blockspergrid_x, blockspergrid_s)

    lap_phi_gpu = cuda.device_array_like(phi_gpu)
    bulk_energy_gpu = cuda.device_array_like(phi_gpu)
    chemical_potential_gpu = cuda.device_array_like(phi_gpu)
    lap_d_phi_gpu = cuda.device_array_like(phi_gpu)

    constant_term = -2.0 * 1e6
    dt_gpu = np.float64(dt)  # Ensure dt is on the device if needed

    for _ in range(nsteps):
        laplacian_gpu[blockspergrid, threadsperblock](phi_gpu, dx, lap_phi_gpu)
        der_bulk_energy_gpu[blockspergrid, threadsperblock](phi_gpu, bulk_energy_gpu)
        cuda.synchronize()  # Ensure both kernels are finished

        # Chemical potential calculation: chemical_potential = bulk_energy - (2.0 * 1e6 * lap_phi)
        cuda.jit(device=True)(_elementwise_subtract_scaled)(bulk_energy_gpu, lap_phi_gpu, constant_term,
                                                            chemical_potential_gpu)
        cuda.synchronize()

        laplacian_gpu[blockspergrid, threadsperblock](chemical_potential_gpu, dx, lap_d_phi_gpu)
        cuda.synchronize()

        # Update phi: phi += dt * lap_d_phi
        cuda.jit(device=True)(_elementwise_add_scaled)(phi_gpu, lap_d_phi_gpu, dt_gpu, phi_gpu)
        cuda.synchronize()  # Ensure update is complete before next iteration

    return phi_gpu


if __name__ == '__main__':
    # Define hyperparameters and call run:
    N = 64
    dx = 10.0  # Or your actual dx
    N_species = 1  # Or your actual N_species
    nsteps = 100
    record_every = 10
    dt = 1e-4
    rng = np.random.default_rng(8)  # For reproducibility
    use_dtype = np.float64

    # Init state as homogenous mixture
    initial_density = 0.01
    densities = np.array([float(initial_density)] * N_species)
    initial_A = 1e-2
    initial_N_peaks = 0
    k = 2 * np.pi * initial_N_peaks / N  # Wavenumber for modulation

    x, y = np.meshgrid(np.arange(N), np.arange(N))
    r = np.sqrt(x ** 2 + y ** 2)
    modulation = initial_A * np.cos(k * r)
    noise = rng.uniform(-1, 1, size=(N_species, N, N))

    if initial_N_peaks == 0:
        random_factor = noise - 0.5
    else:
        random_factor = 1.0 + 0.02 * (noise - 0.5)

    initial_field = densities[:, None, None] * (1.0 + 2.0 * modulation[None, :, :] * random_factor)
    init_phi = initial_field.astype(np.float64)  # Make sure to use the correct dtype

    # Run the simulation:
    recorded_phi = np.zeros((int(nsteps / record_every), N_species, N, N), dtype=use_dtype)
    recorded_phi[0] = init_phi.copy()

    # Transfer initial phi to the GPU
    phi_gpu = cuda.to_device(init_phi.copy())

    for i in range(1, int(nsteps / record_every)):
        phi_gpu = explicit_euler_gpu(phi_gpu, record_every, dt, dx)
        cuda.synchronize()  # Ensure all GPU work is done before transfer

        recorded_phi[i] = phi_gpu.copy_to_host()  # GPU -> CPU for writing

    # Save the final trajectory to a file:
    with open("trajectory.traj", "w") as f:
        for i in range(int(nsteps / record_every)):
            f.write(f"Timestep {i * record_every}:\n")
            temp = recorded_phi[i]
            only_species = temp[0]  # Only 1 species in this simulation, but setup for generalization
            np.savetxt(f, only_species, fmt='%.6f', delimiter=',')
