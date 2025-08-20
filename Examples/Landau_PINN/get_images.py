"""
First analysis script call to ensure the Landau free energy and Euler integrators are appropriately updating the system

I will call this out in each script, but again much of this work comes
from https://github.com/lorenzo-rovigatti/cahn-hilliard and I formatted the input / output to "match". I hope that this
Pythonic version can be used to more rapidly prototype / experiment with. Also, jax enables us to differentiate through
energy phase_field_models, enabling a variety of ML methodologies.
"""
from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [r'solution_species_0.dat']
kwargs_list = [{"cmap": "plasma"}]

plot_all_densities(files1, kwargs_list)

