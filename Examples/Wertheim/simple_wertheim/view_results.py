from pfm.analysis import plot_all_densities, animate, plot_all_energies
from pfm.analysis.multispecies import plot_all_conf

plot_all_densities([r'./cuda/init_0.dat', r'./cuda/last_0.dat', ], [{}, {}])
animate(r'./cuda/trajectory_0.dat')