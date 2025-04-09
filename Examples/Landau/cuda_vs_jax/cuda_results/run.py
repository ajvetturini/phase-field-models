import time
import sys
import subprocess
start = time.time()
subprocess.run(["/home/aj/GitHub/cahn-hilliard/build/bin/ch_2D", "cuda_input.toml"], stdout=subprocess.PIPE)
end = time.time() - start

minutes = int(end // 60)
seconds = int(end % 60)

print(f'CUDA-version of CH finished in: {minutes} min and {seconds} secs')

# This runs in about 49 seconds for 10 million steps