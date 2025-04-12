from pfm.analysis import plot_all_densities, animate, plot_all_energies
from pfm.analysis.multispecies import plot_all_conf

plot_all_conf([r'./cuda/last_0.dat', r'./cuda/last_1.dat', r'./cuda/last_2.dat'])
#animate(r'./cuda/trajectory_2.dat')