import pandas as pd
import os

budgets = [100000, 133000, 167000, 200000, 233000, 267000, 300000, 333000, 367000, 400000]
append_path = os.getcwd()
indices = list(range(1,30))

strings = ['', '_exposed.csv', '_infected.csv', '_susceptible.csv', '_recovered.csv', '_newexposure.csv']


for budget in budgets:
    os.chdir(append_path+'/'+str(budget))
    for x in strings:
        output = pd.read_csv('../../May01_run/100000/May012021PopulationProportion.csv'+x, header=None, sep = ' ')
        last_day = output.shape[1] - 1
        for i in indices:
            columns = list(range(last_day+(i-1)*14+1, last_day+(i)*14+1))
            output[columns] = pd.read_csv('Output_'+str(i)+'.csv'+x, header=None, sep =' ')[list(range(1,15))] 
            output.to_csv('OutputMerged.csv'+x, index=False, header=False, sep = ' ')

print ('Deleting AgeDistrictMatrix.csv file...\n')
os.remove('../../May01_run/AgeDistrictMatrix.csv')


