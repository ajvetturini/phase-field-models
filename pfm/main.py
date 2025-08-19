import toml
from typing import Callable
import time
from pfm.manager import SimulationManager
import jax

def run(config_filepath: str, override_use_jax: bool = False,
        custom_init_fn: Callable = None, custom_energy_fn: Callable = None):
    """ Specify a config file, and optionally a custom init / custom energy function """
    c = toml.load(config_filepath)

    if hasattr(c, 'float_type'):
        if c.get('float_type') == 'float64':
            print('Using float 64 precision')
            jax.config.update("jax_enable_x64", True)

    # After specifying above, import the manager and run:
    manager = SimulationManager(c, custom_energy=custom_energy_fn, custom_initial_condition=custom_init_fn)
    start = time.time()
    manager.run(override_use_jax=override_use_jax)
    end = time.time() - start

    minutes = int(end // 60)
    seconds = int(end % 60)

    print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')
    return minutes, seconds
