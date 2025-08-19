"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all_densities, animate, plot_all_energies

# files1 = [r'./explicit_euler_result_21-00secs/init_0.dat', r'./explicit_euler_result_21-00secs/last_0.dat']
files1 = [r'./spectral_results_30-62secs/init_0.dat', r'./spectral_results_30-62secs/last_0.dat']
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

plot_all_densities(files1, kwargs_list)

animate(r'spectral_results_30-62secs/trajectory_species_0.dat', cmap='plasma', interval=250)
animate(r'explicit_euler_result_21-00secs/trajectory_species_0.dat', cmap='plasma', interval=250)

files = [
    r'./explicit_euler_result_21-00secs/energy.dat',
    r'./spectral_results_30-62secs/energy.dat',
]
labels = [
    'Explit Euler',
    'Spectral Semi-Implicit',
]
kwargs = {"cmap": "plasma", 'labels': labels}
plot_all_energies(files, **kwargs)
