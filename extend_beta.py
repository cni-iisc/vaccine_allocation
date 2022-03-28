import sys
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def extend_beta(days):
    betas = pd.read_csv('AgeDistrictExposureRates.csv', sep = ' ', header=None)
    if betas.shape[1]-1 <days:
        add_days = days - (betas.shape[1]-1)
        for i in range(betas.shape[1], days+1):
            betas[i] = betas[betas.shape[1]-1].tolist()
    betas.to_csv('AgeDistrictExposureRates.csv', header=False, index=False, sep = ' ')

extend_beta(509)
