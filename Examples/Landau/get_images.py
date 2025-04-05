"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all, animate

files1 = [r'./jax_CH/init_0.dat', r'./jax_CH/last_0.dat']
files2 = [r'./cpp_landau/init_0.dat', r'./cpp_landau/last_0.dat']  # C++ Output files
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

plot_all(files1, kwargs_list)
plot_all(files2, kwargs_list)
#animate(r'/home/aj/GitHub/cahn-hilliard/examples/landau/trajectory_0.dat', cmap='plasma')
#animate(r'../Landau/trajectory_0.dat', cmap='plasma')
