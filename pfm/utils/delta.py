"""
The Delta is something like an association strength / bonding volume in the Wertheim-based models
There are specific values that can be over-written / specified via config. The defaults here are the same as
in https://github.com/lorenzo-rovigatti/cahn-hilliard/blob/main/src/utils/Delta.cpp
"""
import numpy as np
from pfm.utils.constants import kb
class Delta:
    def __init__(self, config):
        # Read in the Delta-specific config to get parameters:
        self.T = config.get('T')
        self.salt = config.get('salt', 1.0)
        self.L_stickyend = config.get('sticky_end_length', 6)  # Number of nts in sticky end
        self.delta_H = config.get('deltaH')
        self.delta_S = config.get('deltaS')
        self.bonding_volume = config.get('bonding_volume', 1.6606)  # Units of nm^3

        # Comes from SantaLucia, the 0.368 is the standard value for DNA
        self.salt_entropy_correction_factor = config.get('salt_entropy_correction', 0.368)

        self.delta_S_salt = self.salt_entropy_correction_factor * (self.L_stickyend - 1.0) * np.log(self.salt)
        self.delta_G = self.delta_H - self.T * (self.delta_S + self.delta_S_salt)

        self.k_B = kb
        self.delta = self.bonding_volume * np.exp(-self.delta_G / (self.k_B * self.T))
