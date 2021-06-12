import pandas as pd
import patchsim as sim
import os

budgets = [100000, 133000, 167000, 200000, 233000, 267000, 300000, 333000, 367000, 400000]

append_path = os.getcwd()




for budget in budgets:
    os.chdir(append_path)
    os.chdir(str(budget))

    configs = sim.read_config('AgeDistrictConfigurationPopulationProportional') # assumes no interventions files in input


    replace_dict = {'LoadFile='+configs['LoadFile']: 'LoadFile=Output_0.npy',
            'SaveFile='+configs['SaveFile']:'SaveFile=Output_1.npy',
            'OutputFile='+configs['OutputFile']:'OutputFile=Output_1.csv',
            'LogFile='+configs['LogFile']:'LogFile=Output_1.log',
            'NetworkFile='+configs['NetworkFile']: 'NetworkFile=../AgeDistrictMatrix_nolockdown.csv'}

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



