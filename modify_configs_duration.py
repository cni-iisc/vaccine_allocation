

budgets = [100000, 133000, 167000, 200000, 233000, 267000, 300000, 333000, 367000, 400000]
npis = [1000, 667, 500, 333]

paths = []
for y in npis:
    for x in budgets:
        paths.append('fullrun_'+str(y)+'/'+str(x)+'/')


policies =['AgeDistrictConfigurationPopulationProportional',
    'AgeDistrictConfigurationInvSeroprevalenceProportional',
    'AgeDistrictConfigurationCasesProportional']
path_append = './'

for path in paths:
    for policy in policies:
    
        file = open(path_append+path+policy, 'r')
        
        replaced_content = ""
        
        for line in file:
            
            #stripping line break
            line = line.strip()
            #replacing the texts
            new_line = line.replace("Duration=450", "Duration=509")
            #concatenate the new string and add an end-line break
            replaced_content = replaced_content + new_line + "\n"
        
        file.close()
        #Open file in write mode
        write_file = open(path_append+path+policy, 'w')
        #overwriting the old file contents with the new/replaced content
        write_file.write(replaced_content)
        #close the file
        write_file.close()
