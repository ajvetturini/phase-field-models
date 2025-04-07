"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [r'./jax_long/init_0.dat', r'./jax_long/last_0.dat']
#files2 = [r'./jax_central/init_0.dat', r'./jax_central/last_0.dat']
#files3 = [r'./jax_autodiff/init_0.dat', r'./jax_autodiff/last_0.dat']  # C++ Output files
#files4 = [r'./nocuda_long/init_0.dat', r'./nocuda_long/last_0.dat']  # C++ Output files
files5 = [r'./cuda_long/init_0.dat', r'./cuda_long/last_0.dat']  # C++ Output files
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

plot_all_densities(files1, kwargs_list)
#plot_all_densities(files2, kwargs_list)
#plot_all_densities(files3, kwargs_list)
#plot_all_densities(files4, kwargs_list)
#plot_all_densities(files5, kwargs_list)
animate(r'./jax_long/trajectory_species_0.dat', cmap='plasma', interval=750)
#animate(r'./cuda_long/trajectory_0.dat', cmap='plasma', interval=750)

files = [
    r'./jax_long/energy.dat',
    #r'./jax_central/energy.dat',
    #r'./jax_autodiff/energy.dat',
    #r'./nocuda_long/energy.dat',
    r'./cuda_long/energy.dat',
]
labels = [
    'JAX',  # Fwd differences used in gradient updates
    #'JAX_C',
    #'Jax_Autodiff',
    #'CPP',
    'CUDA',
]
kwargs = {"cmap": "plasma", 'labels': labels}
plot_all_energies(files, **kwargs)
