from pfm import SimulationManager
import toml

c = toml.load(r'input_landau.toml')
SimulationManager(c)
