from pfm.analysis import plot_all_densities, animate, plot_all_energies

#plot_all_densities([r'./cuda/init_0.dat', r'./cuda/last_0.dat', ], [{}, {}])
#animate(r'./cuda/trajectory_0.dat',)
#
plot_all_energies([r'./jax/energy.dat'])
animate(r'./jax/trajectory_species_0.dat',)


