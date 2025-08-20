from pfm.analysis import plot_all_densities, animate, plot_all_energies

#plot_all_densities([r'./jax_explicit_1e8/last_0.dat', r'./jax_spectral/last_0.dat', ], [{}, {}])
#plot_all_energies([r'./jax_explicit_1e8/energy.dat', r'./jax_spectral/energy.dat', r'./cuda/energy_cuda.dat', ])
animate(r'./cuda/trajectory_0.dat', interval=100)
animate(r'./jax_explicit_1e8/trajectory_species_0.dat', interval=100)

#nimate(r'./jax_explicit_1e7/trajectory_species_0.dat', export=False, interval=500)



