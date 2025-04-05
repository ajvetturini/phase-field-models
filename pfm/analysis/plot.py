from matplotlib import pyplot as plt
import numpy as np

def plot_density(fp: str, **kwargs):
    """Plots a single file's worth of data (e.g., a frame of the data)."""
    data = np.loadtxt(fp)
    plt.figure()                      # Create a new figure
    plt.title(fp)                     # Use filename as the title
    im = plt.imshow(data, **kwargs)   # Display data as a heatmap
    if kwargs.get('constant_range', True):
        plt.clim(-1, 1)  # Set range
    plt.colorbar(im, )                  # Add colorbar
    plt.show()

def plot_all_densities(list_of_fps: list, list_of_kwargs: list):
    """Plots multiple files, each with its corresponding keyword arguments."""
    for fp, kwargs in zip(list_of_fps, list_of_kwargs):
        plot_density(fp, **kwargs)  # Correctly unpack kwargs


def _get_energy(fp: str):
    data = np.loadtxt(fp)
    time, avg_energy, avg_density = data[:, 0], data[:, 1], data[:, 2]
    return time, avg_energy, avg_density


def plot_energy(fp: str, **kwargs):
    """ Plots the average free energy and average mass density against time from a simulation """
    t, e, rho = _get_energy(fp)

    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Simulation Energy and Mass Density vs. t')

    axs[0].plot(t, e)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Average Free Energy')
    axs[0].set_title('Average Free Energy vs. t')

    axs[1].plot(t, rho)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Average Mass Density')
    axs[1].set_title('Average Mass Density vs. t')

    plt.show()

def plot_all_energies(list_of_fps: list, **kwargs):
    """ Plots multiple files, each with its corresponding keyword arguments."""

    all_t, all_e, all_rho = [], [], []
    for fp in list_of_fps:
        nt, ne, nr = _get_energy(fp)
        all_t.append(nt)
        all_e.append(ne)
        all_rho.append(nr)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    all_labels = kwargs.get('labels', [])

    for ct, (i, j) in enumerate(zip(all_t, all_e)):
        if all_labels:
            axs[0].plot(i, j, label=all_labels[ct])
        else:
            axs[0].plot(i, j)
        axs[0].set_xlabel('Time steps')
        axs[0].set_ylabel('Average Free Energy')
        axs[0].legend()  # Add legend to the first subplot

    for ct, (i, j) in enumerate(zip(all_t, all_rho)):
        if all_labels:
            axs[1].plot(i, j, label=all_labels[ct])
        else:
            axs[1].plot(i, j)
        axs[1].set_xlabel('Time steps')
        axs[1].set_ylabel('Average Mass Density')
        axs[1].ticklabel_format(axis='y', style='sci')
        axs[1].set_ylim(-0.1, 0.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()