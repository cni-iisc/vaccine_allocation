import pandas as pd

budgets = [100000, 133000, 167000, 200000, 233000, 267000, 300000, 333000, 367000, 400000]
npis = [1000, 667, 500, 333]

paths = []
for y in npis:
    for x in budgets:
        paths.append('fullrun_'+str(y)+'/'+str(x)+'/')


policies =['AgeDistrictVaxFilePopulationProportional.csv',
    'AgeDistrictVaxFileSeroprevalenceProportional.csv',
    'AgeDistrictVaxFileCasesProportional.csv']
path_append = './'

for path in paths:
    for policy in policies:
        df = pd.read_csv(path_append+path+policy, sep = ' ', header = None)
        df1 = df.loc[df[0]==446]
        for i in range(0,60):
            df1[0] = 446+1+i
            df = df.append(df1, ignore_index = True)
        df.to_csv(path_append+path+policy, sep = ' ', index=False, header=False)
        