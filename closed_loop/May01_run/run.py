import sys
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

import patchsim as sim


os.chdir('./100000/')    
configs = sim.read_config('AgeDistrictConfigurationPopulationProportional') # assumes no interventions files in input
numberofdays = int(configs['Duration'])
outputfile = configs['OutputFile']
print("Configurations used:")
print(configs)
sim.run_disease_simulation(configs,write_epi=True)

