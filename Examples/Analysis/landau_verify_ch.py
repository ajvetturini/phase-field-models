"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all, animate

files = [r'../Landau/init_0.dat', r'../Landau/last_0.dat']
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

#plot_all(files, kwargs_list)
animate(r'../Landau/trajectory_0.dat', cmap='plasma')
