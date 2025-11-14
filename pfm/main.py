import toml
from typing import Callable
import time
import os

def run(config_filepath: str, override_use_jax: bool = False,
        custom_init_fn: Callable = None, custom_energy_fn: Callable = None):
    """ Specify a config file, and optionally a custom init / custom energy function """
    c = toml.load(config_filepath)

    # Before importing JAX, check for relevent environment variables in the config:
    if "CUDA_VISIBLE_DEVICES" in c:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(c["CUDA_VISIBLE_DEVICES"])

    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in c:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(c["XLA_PYTHON_CLIENT_MEM_FRACTION"])

    # Specify float type prior to import jax.numpy in the SimulationManager
    import jax
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
    return minutes, seconds
