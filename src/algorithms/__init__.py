from src.algorithms.other_discretizations import *
from src.algorithms.vi import *
from src.algorithms.algorithms import *


ALGORITHMS = {'pgd': ParticleML,
              'MPD_Nest': MPD_Nesterov,
              'MPD': MPD_Proposed,
              'MPD_NC': MPD_NCTwo, # MPD With No Corrections.
              'abp': ABP,
              'sr': ShortRun,
              'vi': VI,}