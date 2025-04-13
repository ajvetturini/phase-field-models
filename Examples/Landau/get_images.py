"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [r'./init_0.dat', r'./last_0.dat']
#files2 = [r'./autodiff_results/init_0.dat', r'./autodiff_results/last_0.dat']
#files3 = [r'./cuda_results/init_0.dat', r'./cuda_results/last_0.dat']
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

plot_all_densities(files1, kwargs_list)
#plot_all_densities(files2, kwargs_list)
#plot_all_densities(files3, kwargs_list)

animate(r'./trajectory_species_0.dat', cmap='plasma', interval=250)

files = [
    r'./energy.dat',
    #r'./autodiff_results/energy.dat',
    #r'./cuda_results/energy.dat',
]
labels = [
    'JAX',
    #'Autodiff (JAX)',
    #'CUDA',
]
kwargs = {"cmap": "plasma", 'labels': labels}
plot_all_energies(files, **kwargs)
