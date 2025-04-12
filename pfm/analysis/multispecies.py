# Various utilities for compiling multi-species results into single viewables
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_all_conf(conf_files: list, **kwargs):
    """
    Reads data from a list of CONF files, normalizes it and show image
    """
    rho_max = 0
    data = []
    for f in conf_files:
        try:
            loaded_data = np.loadtxt(f)
            data.append(loaded_data)
            data_max = np.max(loaded_data)
            if data_max > rho_max:
                rho_max = data_max
        except FileNotFoundError:
            print(f"Error: File not found: {f}")
            return
        except Exception as e:
            print(f"Error reading file {f}: {e}")
            return

    if not data:
        print("Error: No valid data loaded from the provided files.", file=sys.stderr)
        return

    normalized_data = [d / rho_max for d in data]
    stacked_data = (np.dstack(normalized_data) * 255.999).astype(np.uint8)
    new_image = Image.fromarray(stacked_data)
    plt.imshow(new_image)
    plt.show()
