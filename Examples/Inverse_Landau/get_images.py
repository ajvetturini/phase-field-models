from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [r'init_0.dat', r'last_0.dat']
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}]

#plot_all_densities(files1, kwargs_list)
animate(r'trajectory_species_0.dat', cmap='plasma', interval=250)

files = [
    r'energy.dat',
]
labels = [
    'total_energy'
]
kwargs = {"cmap": "plasma", 'labels': labels}
#plot_all_energies(files, **kwargs)
