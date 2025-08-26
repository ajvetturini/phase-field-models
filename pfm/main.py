import toml
from typing import Callable
import time
import jax

def run(config_filepath: str, override_use_jax: bool = False,
        custom_init_fn: Callable = None, custom_energy_fn: Callable = None):
    """ Specify a config file, and optionally a custom init / custom energy function """
    c = toml.load(config_filepath)

    if c.get('float_type', 'float32') == 'float64':
        jax.config.update("jax_enable_x64", True)

    # After specifying above, import the manager and run for float64 safety:
    from pfm.manager import SimulationManager
    start = time.time()
    manager = SimulationManager(c, custom_energy=custom_energy_fn, custom_initial_condition=custom_init_fn)
    manager.run(override_use_jax=override_use_jax)

    end = time.time() - start
    minutes = int(end // 60)
    seconds = int(end % 60)

    print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')
    return minutes, seconds
