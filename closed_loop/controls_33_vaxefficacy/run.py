import sys
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from datetime import date, timedelta

import patchsim as sim

def get_districts():
    return pd.read_csv('../../../seroprevalance_modified.csv')['Unit'].tolist()



def get_latest_cir_factor():
    testing_data = pd.read_excel('../../../target_curves/testing.xlsx')
    ref_value = np.average(testing_data['Tests'].values[0:10])

    locations = [list(range(398,402))]
    cir_factors = []
    for i in range(0,len(locations)):
        temp = 0
        for j in range(0,len(locations[i])):
            temp = temp + testing_data.loc[testing_data['date_index']==locations[i][j],'Tests'].item()
        temp = temp/len(locations[i])
        cir_factors.append(temp/ref_value)
    return cir_factors[0]

def prepare_district_simulation_plots(outputfile, start_index_plot, days_plot):
    jump_values = 7

    districts = get_districts()
    simulation = pd.read_csv(outputfile, header=None, sep=' ')
    district_simulation = pd.DataFrame(columns=['DistrictName']+(list(range(0,days_plot))))
    district_simulation['DistrictName'] = districts
    sero_data = pd.read_csv('../../../seroprevalance_modified.csv')
    for x in districts:
        cir_factor = get_latest_cir_factor()
        cir = sero_data.loc[sero_data['Unit']==x, 'CIR'].item()/cir_factor
        for j in range(days_plot):
            i = j + start_index_plot
            district_simulation.at[district_simulation['DistrictName']==x, j] = \
                (np.sum(simulation.loc[simulation[0].str.contains(x), i+1].tolist())/cir).astype(int)
    return district_simulation

def compute_mean_active_cases(start_week_index):
    if start_week_index ==1:
        df = prepare_district_simulation_plots('../../May01_run_33_vaxefficacy/100000/May012021PopulationProportion.csv_infected.csv', 195, 14)
    else:
        df = prepare_district_simulation_plots('Output_'+str(start_week_index-1)+'.csv_infected.csv', 0, 14)


    return (float(np.average(np.sum(df[list(range(0, 14))].values, axis = 0))))


def modify_matrix_line(npi):
    configs = sim.read_config('AgeDistrictConfigurationPopulationProportional') # assumes no interventions files in input
    old_line_text = 'NetworkFile='+configs['NetworkFile']
    new_line_text = 'NetworkFile=../Matrix_'+str(npi)+'.csv'
    file = open('AgeDistrictConfigurationPopulationProportional', 'r')

    replaced_content = ""

    for line in file:

        #stripping line break
        line = line.strip()
        #replacing the texts
        new_line = line.replace(old_line_text, new_line_text)
        #concatenate the new string and add an end-line break
        replaced_content = replaced_content + new_line + "\n"

    file.close()
    #Open file in write mode
    write_file = open('AgeDistrictConfigurationPopulationProportional', 'w')
    #overwriting the old file contents with the new/replaced content
    write_file.write(replaced_content)
    #close the file
    write_file.close()

def modify_output_lines(start_week_index):

    replace_dict = {'LoadFile=Output_'+str(start_week_index-2)+'.npy': 'LoadFile=Output_'+str(start_week_index-1)+'.npy',
            'SaveFile=Output_'+str(start_week_index-1)+'.npy':'SaveFile=Output_'+str(start_week_index)+'.npy',
            'OutputFile=Output_'+str(start_week_index-1)+'.csv':'OutputFile=Output_'+str(start_week_index)+'.csv',
            'LogFile=Output_'+str(start_week_index-1)+'.log':'LogFile=Output_'+str(start_week_index)+'.log'} 

    for key in replace_dict:
        input_string = key
        output_string = replace_dict[key]

        file = open('AgeDistrictConfigurationPopulationProportional', 'r')

        replaced_content = ""

        for line in file:

            #stripping line break
            line = line.strip()
            #replacing the texts
            new_line = line.replace(input_string, output_string)
            #concatenate the new string and add an end-line break
            replaced_content = replaced_content + new_line + "\n"

        file.close()
        #Open file in write mode
        write_file = open('AgeDistrictConfigurationPopulationProportional', 'w')
        #overwriting the old file contents with the new/replaced content
        write_file.write(replaced_content)
        #close the file
        write_file.close()

def modify_config(budget, start_week_index):
    mean_active = compute_mean_active_cases(start_week_index)
    if mean_active <= 20000:
        npi_array.append(1000)
        modify_matrix_line(1000)
    elif mean_active >=20001 and mean_active <=100000:
        npi_array.append(667)
        modify_matrix_line(667)
    elif mean_active >=100001 and mean_active <=200000:
        npi_array.append(500)
        modify_matrix_line(500)
    elif mean_active >=200001:
        npi_array.append(333)
        modify_matrix_line(333)


ref_date = date(2021,5,2)
dates = [ref_date]
total_weeks = 20
for i in range(0,total_weeks-1):
    dates.append(ref_date+timedelta(days=(i+1)*14))

start_indices = list(range(1,total_weeks))
budgets = [100000, 133000, 167000, 200000, 233000, 267000,300000, 333000, 367000, 400000]

present_directory = os.getcwd()
for budget in budgets:
    npi_array = []
    os.chdir(present_directory)
    os.chdir(str(budget))    
    for i in start_indices: 
        modify_config(budget, i)
        if i>=2:
            modify_output_lines(i)

        configs = sim.read_config('AgeDistrictConfigurationPopulationProportional') # assumes no interventions files in input
        numberofdays = int(configs['Duration'])
        outputfile = configs['OutputFile']
        print("Configurations used:")
        print(configs)
        sim.run_disease_simulation(configs,write_epi=True)
    np.savetxt('npi_array.csv', npi_array, delimiter=',')
    print(budget, 'is complete ...')
