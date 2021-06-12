# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:10:22 2021

@author: sarat
"""
import sys
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

sys.path.append('../')
import patchsim as sim

daily_budgets = [100000, 133000]# 167000, 200000, 233000, 267000, 300000, 333000, 367000, 400000]

present_directory = os.getcwd()
for daily_budget in daily_budgets:
    os.chdir(present_directory)
    os.chdir('./'+str(daily_budget)) 
# run scenario 1
    configs = sim.read_config('AgeDistrictConfigurationPopulationProportional') # assumes no interventions files in input
    numberofdays = int(configs['Duration'])
    outputfile = configs['OutputFile']
    print("Configurations used:")
    print(configs)
    sim.run_disease_simulation(configs,write_epi=True)
    
    # run scenario 2
    configs = sim.read_config('AgeDistrictConfigurationInvSeroprevalenceProportional') # assumes no interventions files in input
    numberofdays = int(configs['Duration'])
    outputfile = configs['OutputFile']
    print("Configurations used:")
    print(configs)
    sim.run_disease_simulation(configs,write_epi=True)
    
    # run scenario 3
    configs = sim.read_config('AgeDistrictConfigurationCasesProportional') # assumes no interventions files in input
    numberofdays = int(configs['Duration'])
    outputfile = configs['OutputFile']
    print("Configurations used:")
    print(configs)
    sim.run_disease_simulation(configs,write_epi=True)

    print(str(daily_budget), 'complete ...')
