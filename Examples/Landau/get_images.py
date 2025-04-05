"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [r'./jax_CH/init_0.dat', r'./jax_CH/last_0.dat']
files2 = [r'./cpp_landau/init_0.dat', r'./cpp_landau/last_0.dat']  # C++ Output files
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

#plot_all_densities(files1, kwargs_list)
#plot_all_densities(files2, kwargs_list)
#animate(r'./jax_CH/trajectory_0.dat', cmap='plasma', interval=750)
#animate(r'./cpp_landau/trajectory_0.dat', cmap='plasma', interval=750)

files = [r'./jax_CH/energy.dat', r'./cpp_landau/energy.dat']
labels = ['JAX', 'CPP']
kwargs = {"cmap": "plasma", 'labels': labels}
plot_all_energies(files, **kwargs)
