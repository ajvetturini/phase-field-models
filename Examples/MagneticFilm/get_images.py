from pfm.analysis import plot_all_densities, animate, plot_energy
files2 = [r'last_0.dat']
kwargs_list = [{"cmap": "grey"}]

plot_all_densities(files2, kwargs_list)
#animate('trajectory_0.dat')
plot_energy('energy.dat', use_timesteps=True)

