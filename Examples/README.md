These folders contain the input files used to generate the data in the figures [here](../README.md).

The Landau, Saleh Wertheim, and Simple Wertheim also contain their initial condition used. Note, I generated the saleh-wertheim data using randomized initial conditions as I didn't have the functionality to read in multi-species initial conditions (and these simulations take tens of hours to run). However, the simulation is "consistent" in that you will see some variation of the GIF (i.e., a similar average system energy).

The MagneticFilm shows how you can implement a custom free energy functional + initial condition to the SimulationManager. Make sure that you are careful with where you import the float64 (i.e., either needs to be environment variable OR use the config update PRIOR to importing jax.numpy)

Lastly, each input file needs to have specific paths specified (i.e., where to write files), so be careful if you plan to re-run these inputs specifically!