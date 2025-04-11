from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [r'./jax/init_0.dat', r'./jax/last_0.dat']
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

plot_all_densities(files1, kwargs_list)

animate(r'./jax/trajectory_0.dat', cmap='plasma', interval=750)

files = [
    r'./jax/energy.dat',
]
labels = [
    'jax',
]
kwargs = {"cmap": "plasma", 'labels': labels}
plot_all_energies(files, **kwargs)
