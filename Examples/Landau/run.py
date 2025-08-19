from pfm import run
import time

start = time.time()
run(r'jax_input.toml', True)
end = time.time() - start
print(f"Execution time: {end:.2f} seconds")