import os
import shutil
import zipfile

budgets = [100000, 133000, 167000, 200000, 233000, 267000, 300000, 333000, 367000, 400000]

for budget in budgets:
    shutil.copyfile('May01_run/100000/May012021PopulationProportion.npy', 'controls/'+str(budget)+'/Output_0.npy')
    shutil.copyfile('May01_run/100000/May012021PopulationProportion.npy_wanearray.npy', 'controls/'+str(budget)+'/Output_0.npy_wanearray.npy')


