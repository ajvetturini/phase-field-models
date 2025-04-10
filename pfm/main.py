# This main script is useful so that a CUDA device can be specified prior to JAX utils being imported
# You can also directly import the SimulationManager (but beware of setting CUDA device number)

import toml
import os
from typing import Callable
import time
def run(config_filepath: str, custom_init_fn: Callable = None, custom_energy_fn: Callable = None):
    """ Specify a config file, and optionally a custom init / custom energy function """
    c = toml.load(config_filepath)

    if 'gpu_device_number' not in c:
        print('Accelerator not specified, if using GPU / TPU then the 0th device will likely be used.')
    else:
        device_number = c.get('gpu_device_number')
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_number

    # After specifying above, import the manager and run:
    from pfm.manager import SimulationManager
    manager = SimulationManager(c, custom_energy=custom_energy_fn, custom_initial_condition=custom_init_fn)
    start = time.time()
    manager.run()
    end = time.time() - start

    minutes = int(end // 60)
    seconds = int(end % 60)

    print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')
