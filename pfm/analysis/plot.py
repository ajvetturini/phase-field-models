from matplotlib import pyplot as plt
import numpy as np

def plot(fp: str, **kwargs):
    """Plots a single file's worth of data (e.g., a frame of the data)."""
    data = np.loadtxt(fp)
    plt.figure()                      # Create a new figure
    plt.title(fp)                     # Use filename as the title
    im = plt.imshow(data, **kwargs)   # Display data as a heatmap
    plt.colorbar(im)                  # Add colorbar
    plt.show()

def plot_all(list_of_fps: list, list_of_kwargs: list):
    """Plots multiple files, each with its corresponding keyword arguments."""
    for fp, kwargs in zip(list_of_fps, list_of_kwargs):
        plot(fp, **kwargs)  # Correctly unpack kwargs
