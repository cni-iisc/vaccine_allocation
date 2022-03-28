import pandas as pd

betas = pd.read_csv('../AgeDistrictExposureRates.csv', sep = ' ', header=None)

start_index_may2 = 203
indices = range(start_index_may2+1,betas.shape[1])
num_agedist = betas.shape[0]

factor = 1.1

for j in indices:
    for i in range(num_agedist):
        betas.at[i,j] = betas.loc[i,j]*factor
betas.to_csv('vaccine_allocation_beta+10/AgeDistrictExposureRates.csv', sep = ' ', header=False, index=False)


factor = 0.9

for j in indices:
    for i in range(num_agedist):
        betas.at[i,j] = betas.loc[i,j]*factor
betas.to_csv('vaccine_allocation_beta-10/AgeDistrictExposureRates.csv', sep = ' ', header=False, index=False)