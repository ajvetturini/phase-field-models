from pfm.analysis import plot_all_densities, animate, plot_all_energies

files1 = [
    r'cuda/last_0.dat',
    r'nocuda/last_0.dat',
    r'euler_autodiff/last_0.dat',
    r'euler_noautodiff/last_0.dat',
    r'spectral_autodiff/last_0.dat',
    r'spectral_noautodiff/last_0.dat',
]
kwargs_list = [{"cmap": "plasma"}, {"cmap": "plasma"}, {"cmap": "plasma"}, {"cmap": "plasma"}, {"cmap": "plasma"}, {"cmap": "plasma"}, {"cmap": "plasma"}]

plot_all_densities(files1, kwargs_list)
#animate(r'trajectory_species_0.dat', cmap='plasma', interval=250)

files = [
    r'cuda/energy.dat',
    r'nocuda/energy.dat',
    r'euler_autodiff/energy.dat',
    r'euler_noautodiff/energy.dat',
    r'spectral_autodiff/energy.dat',
    r'spectral_noautodiff/energy.dat',
]
labels = [
    'cuda',
    'nocuda',
    'euler_autodiff',
    'euler_noautodiff',
    'spectral_autodiff',
    'spectral_noautodiff',
]
kwargs = {"cmap": "plasma", 'labels': labels}
plot_all_energies(files, **kwargs)
