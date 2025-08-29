from pfm.analysis import plot_all_densities, animate, plot_all_energies

plot_all_densities([r'./euler_noautodiff/last_0.dat', r'./cuda/last_0.dat', ], [{}, {}])
# plot_all_energies([r'./euler_noautodiff/energy.dat', r'./cuda/energy.dat', ])
#animate(r'./cuda/trajectory_0.dat', interval=100, smoothen_animation=False)
#animate(r'./euler_noautodiff/trajectory_species_0.dat', interval=100, smoothen_animation=False)
#animate(r'./spectral_f32/trajectory_species_0.dat', interval=100, smoothen_animation=False)

#nimate(r'./jax_explicit_1e7/trajectory_species_0.dat', export=False, interval=500)



