from pfm.analysis import plot_all_densities, animate, plot_all_energies

#plot_all_densities([r'./cuda/init_0.dat', r'./cuda/last_0.dat', ], [{}, {}])
#animate(r'./cuda/trajectory_0.dat',)

labels = ['Autodiff (F64)', 'JAX (F32)', 'CUDA (F64)', ]
#plot_all_energies([r'./jax/energy_autodiff.dat', r'./jax/energy_float32.dat', r'./cuda/energy_cuda.dat'], labels=labels)
animate(r'./jax/trajectory_autodiff.dat', export=True, interval=500)
#animate(r'./jax/trajectory_species_0_float32.dat', export=True, interval=500)
animate(r'./cuda/trajectory_0.dat', export=True, interval=500)


