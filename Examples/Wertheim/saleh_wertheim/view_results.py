from pfm.analysis import plot_multispecies_densities, animate_multispecies

c0 = r'C:\Users\Anthony\Documents\GitHub\phase-field-models\Examples\Wertheim\saleh_wertheim\cuda\trajectory_0.dat'
c1 = r'C:\Users\Anthony\Documents\GitHub\phase-field-models\Examples\Wertheim\saleh_wertheim\cuda\trajectory_1.dat'
c2 = r'C:\Users\Anthony\Documents\GitHub\phase-field-models\Examples\Wertheim\saleh_wertheim\cuda\trajectory_2.dat'

#plot_multispecies_densities([c0, c1, c2])
animate_multispecies([c0, c1, c2])
