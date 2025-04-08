from pfm import SimulationManager
import toml
import time

c = toml.load(r'input_wertheim_simple.toml')
manager = SimulationManager(c)
start = time.time()
manager.run()
end = time.time() - start

minutes = int(end // 60)
seconds = int(end % 60)

print(f'JAX-version of CH finished in: {minutes} min and {seconds} secs')