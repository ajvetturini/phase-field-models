from pfm import run
import time
import subprocess
import shutil
import os



def run_pfm(toml_file):
    start = time.time()
    run(toml_file, True)
    end = time.time() - start
    return end

#time1 = run_pfm('euler_autodiff.toml')
time2 = run_pfm('euler_noautodiff.toml')
#time3 = run_pfm('spectral_noautodiff.toml')
#time4 = run_pfm('spectral_autodiff.toml')
#all_times_to_write = [('euler_autodiff', time1), ('euler_noautodiff', time2), ('spectral_noautodiff', time3), ('spectral_autodiff', time4)]

def time_ch_and_move(command, move_dir):
    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end = time.time() - start

    # Move files:
    shutil.move('energy.dat', os.path.join(move_dir, 'energy.dat'))
    shutil.move('init_0.dat', os.path.join(move_dir, 'init_0.dat'))
    shutil.move('init_density.dat', os.path.join(move_dir, 'init_density.dat'))
    shutil.move('last_0.dat', os.path.join(move_dir, 'last_0.dat'))
    shutil.move('last_density.dat', os.path.join(move_dir, 'last_density.dat'))
    shutil.move('trajectory_0.dat', os.path.join(move_dir, 'trajectory_0.dat'))
    return end



#command1 = ["/mnt/nvme/home/avetturi/cahn-hilliard/build/bin/ch_2D", "/mnt/nvme/home/avetturi/phase-field-models/Examples/Landau/cuda/input_cuda.toml"]
#command2 = ["/mnt/nvme/home/avetturi/cahn-hilliard/build/bin/ch_2D", "/mnt/nvme/home/avetturi/phase-field-models/Examples/Landau/nocuda/input_nocuda.toml"]

#time_to_run = time_ch_and_move(command1, 'cuda')
#all_times_to_write.append(('cuda', time_to_run))
#time_to_run = time_ch_and_move(command2, 'nocuda')
#all_times_to_write.append(('nocuda', time_to_run))


# Write out to txt file the timed results:
'''with open('landau_all_timed_results.txt', 'w') as f:
    f.write('Landau Results (times reported in seconds)\n')  
    for result in all_times_to_write:
        name, final_time = result
        f.write(f'{name} | {final_time}\n') '''