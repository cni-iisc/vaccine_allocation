import sys
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def get_districts():
    return pd.read_csv('../vaccine_allocation/seroprevalence_modified.csv')['Unit'].tolist()

def get_beta_districts(beta):
    districts = get_districts()
    districts_beta = pd.DataFrame(columns = ['district', 'beta'])
    for x in districts:
        new_row = {'district':x, 'beta':beta.loc[beta[0].str.contains(x),2].values.tolist()[0] }
        districts_beta = districts_beta.append(new_row, ignore_index=True)
    return districts_beta

def find_slope_from_regression(data):
    param0 = [1,data[0]]
    n = len(data)
    def obj_fn(param):
        return (param[1]+param[0]*np.arange(0,n)) - data    
    return least_squares(obj_fn,param0).x[0]

def get_diff_sum(start_index, start_index_simulation, start_index_calibration, days_calibration, district_frame, outputfile, duration):
    districts = get_districts()
    return_diff = pd.DataFrame(columns = ['district', 'diff'])
    district_simulation = prepare_district_simulation(outputfile, start_index_calibration, days_calibration, duration)
    for x in districts:
        diff = np.average(district_frame.loc[district_frame['DistrictName']==x].values.tolist()[0][start_index_simulation+1:start_index_simulation+1+days_calibration]) - np.average(district_simulation.loc[district_simulation['DistrictName']==x].values.tolist()[0][1::])

        new_row = {'district':x, 'diff': diff}
        return_diff = return_diff.append(new_row, ignore_index=True)
    return return_diff

def prepare_district_frame(start_index_calibration, days_calibration, start_index):
    districts = get_districts()
    smooth_days = 7
    inf = pd.read_csv('../vaccine_allocation/target_curves/tinf_df.csv')
    
    bengaluru_zones = ['Bengaluru-Urban', 'BBMP-Bommanahalli', 'BBMP-Dasarahalli', 'BBMP-East', \
                'BBMP-Mahadevpura','BBMP-RR-Nagar', 'BBMP-South', 'BBMP-West', 'BBMP-Yelahanka']
    bbmp_zone_proportion = [ 3608,4645, 2275, 7605, 5956, 5640, 8709, 9011, 3608]  # read on 17th april from covid bulletin of BBMP. assume urban is same as yelahanka
    bbmp_zone_proportion = bbmp_zone_proportion/np.sum(bbmp_zone_proportion)

    
    district_frame = pd.DataFrame(columns=['DistrictName']+(list(range(0,days_calibration))))
    district_frame['DistrictName'] = districts

    for x in districts:
        if x in bengaluru_zones:
            district_index = inf[inf['Unit']=='Bengaluru-Urban'].index.item()
        else:
            district_index = district_index = inf[inf['Unit']==x].index.item()
        i_data = np.diff(inf.iloc[district_index,2:].values[start_index-smooth_days::].astype(int))
        i_data_average = (np.convolve(i_data, np.ones(smooth_days))/smooth_days)[smooth_days-1::].astype(int).tolist()[start_index_calibration:start_index_calibration+days_calibration]

        for i in range(days_calibration):
            if x in bengaluru_zones:
                zone_index = bengaluru_zones.index(x)
                district_frame.at[district_frame['DistrictName']==x,i] = int(i_data_average[i]*bbmp_zone_proportion[zone_index])
            else:
                district_frame.at[district_frame['DistrictName']==x,i] = int(i_data_average[i])
    return district_frame     

def prepare_district_simulation(outputfile, start_index_calibration, days_calibration, duration):
    jump_values = 7
    districts = get_districts()
    simulation = pd.read_csv(outputfile, header=None, sep=' ')
    district_simulation = pd.DataFrame(columns=['DistrictName']+(list(range(0,days_calibration))))
    district_simulation['DistrictName'] = districts
    sero_data = pd.read_csv('../vaccine_allocation/seroprevalence_modified.csv')
    cir_factors = get_cir_factors(duration)
    for x in districts:
        for i in range(days_calibration):
            cir_index = int(np.minimum(1,  i/jump_values))
            cir = sero_data.loc[sero_data['Unit']==x, 'CIR'].item()/cir_factors[cir_index]
            district_simulation.at[district_simulation['DistrictName']==x, i] = \
                (np.sum(simulation.loc[simulation[0].str.contains(x), start_index_calibration+i+1].tolist())/cir).astype(int)
    return district_simulation

def prepare_district_simulation_plots(outputfile, days_plot):
    jump_values = 7
    
    districts = get_districts()
    simulation = pd.read_csv(outputfile, header=None, sep=' ')
    district_simulation = pd.DataFrame(columns=['DistrictName']+(list(range(0,days_plot))))
    district_simulation['DistrictName'] = districts
    sero_data = pd.read_csv('../vaccine_allocation/seroprevalence_modified.csv')
    for x in districts:
        for i in range(days_plot):
            if i<=140:
                cir = sero_data.loc[sero_data['Unit']==x, 'CIR'].item()
            else:
               if i>=141 and i<=155:
                   duration = 'March1-15'
                   j = i - 141
               elif i>=156 and i<=178:
                   duration = 'March16-April7'
                   j = i - 156
               elif i>=179 and i<=193:
                   duration = 'April8-22'
                   j = i - 179
               elif i>=194:
                   duration = 'April23-present'
                   j = i - 194
               
               cir_factors = get_cir_factors(duration)
       
               cir_index = int(np.minimum(1,  j/jump_values))

               cir = sero_data.loc[sero_data['Unit']==x, 'CIR'].item()/cir_factors[cir_index]
            district_simulation.at[district_simulation['DistrictName']==x, i] = \
                (np.sum(simulation.loc[simulation[0].str.contains(x), i+1].tolist())/cir).astype(int)
    return district_simulation

def get_cir_factors(duration):
    testing_data = pd.read_excel('../vaccine_allocation/target_curves/testing.xlsx')
    ref_value = np.average(testing_data['Tests'].values[0:10])

    if duration == 'March1-15':
        locations = [list(range(345,352)),list(range(352,360))]
    elif duration == 'March16-31':
        locations = [list(range(360,367)),list(range(367,376))]
    elif duration == 'March16-April7':
        locations = [list(range(360,367)),list(range(367,376)), list(range(376,383))]
    elif duration == 'April1-15':
        locations = [list(range(376,383)),list(range(383,391))]
    elif duration == 'April8-22':
        locations = [list(range(383,391)),list(range(391,398))]
    elif duration == "April23-present":
        locations = [list(range(391,398)),list(range(398,402))]


    cir_factors = []
    for i in range(0,len(locations)):
        temp = 0
        for j in range(0,len(locations[i])):
            temp = temp + testing_data.loc[testing_data['date_index']==locations[i][j],'Tests'].item()
        temp = temp/len(locations[i])
        cir_factors.append(temp/ref_value)
    return cir_factors




def generate_plots_comparison(outputfile1_case1,outputfile2_case1,outputfile3_case1, start_index, plot_days_data, plot_days_simulation, smooth_days, sub_dir, start_offset):
    delay = 4
    sero_data = pd.read_csv('../vaccine_allocation/seroprevalence_modified.csv')
    district_frame = prepare_district_frame(0,plot_days_data, start_index)
    district_simulation_1_case1 = prepare_district_simulation_plots(outputfile1_case1,plot_days_simulation)
    district_simulation_2_case1 = prepare_district_simulation_plots(outputfile2_case1,plot_days_simulation)
    district_simulation_3_case1 = prepare_district_simulation_plots(outputfile3_case1,plot_days_simulation)

    districts = get_districts()
    if start_offset == 20:
        x_indices_temp = [0, 21, 51, 82, 113, 141, 172, 202, 233, 263, 294, 325, 355, 386, 416, 447, 478, 505]
        x_indices = [a-(start_offset+1) for a in x_indices_temp][1::]
        x_labels = ['01 Nov 2020', '01 Dec 2020', '01 Jan 2021', '01 Feb 2021', '01 Mar 2021', '01 Apr 2021', '01 May 2021', '01 Jun 2021', '01 Jul 2021', '01 Aug 2021', '01 Sep 2021', '01 Oct 2021', '01 Nov 2021', '01 Dec 2021',  '01 Jan 2022', '01 Feb 2022', '28 Feb 2022']
    else:
        x_indices = [0, 21, 51, 82, 113, 141, 172, 202, 233, 263, 294, 324]
        x_labels = ['11 Oct 2020', '01 Nov 2020', '01 Dec 2020', '01 Jan 2021', '01 Feb 2021', '01 Mar 2021', '01 Apr 2021', '01 May 2021', '01 Jun 2021', '01 Jul 2021', '01 Aug 2021', '31 Aug 2021']
    karnataka_curve1_case1 = np.full(plot_days_simulation, 0)[start_offset+delay::]
    karnataka_curve2_case1 = np.full(plot_days_simulation, 0)[start_offset+delay::]
    karnataka_curve3_case1 = np.full(plot_days_simulation, 0)[start_offset+delay::]
    
    karnataka_data = np.full(plot_days_data, 0)[start_offset+delay::]
    
    plt.rcParams['figure.figsize'] = [18, 10]
    plt.rcParams['font.size'] = 26

    for x in districts:
        curve1_case1= district_simulation_1_case1.loc[district_simulation_1_case1['DistrictName']==x].values[0][1::][start_offset+delay::]
        curve2_case1= district_simulation_2_case1.loc[district_simulation_2_case1['DistrictName']==x].values[0][1::][start_offset+delay::]
        curve3_case1= district_simulation_3_case1.loc[district_simulation_3_case1['DistrictName']==x].values[0][1::][start_offset+delay::]
        curve_data = district_frame.loc[district_frame['DistrictName']==x].values[0][1::][start_offset+delay::]
        
        karnataka_curve1_case1 =  karnataka_curve1_case1 + curve1_case1
        karnataka_curve2_case1 =  karnataka_curve2_case1 + curve2_case1
        karnataka_curve3_case1 =  karnataka_curve3_case1 + curve3_case1
        
        
        karnataka_data = karnataka_data + curve_data

      
    plt.plot((karnataka_curve1_case1), 'b',label='Baseline')
    plt.plot((karnataka_curve2_case1), 'g',label=label_names[parameter_case] + ' + 10%')
    plt.plot((karnataka_curve3_case1), 'c',label=label_names[parameter_case] + ' - 10%')
        
    plt.plot((karnataka_data), 'r',label='Reported cases')
    plt.xticks(x_indices, x_labels, rotation='vertical')
    plt.grid(True)
    plt.ylabel('Daily cases')
    plt.title('Karnataka -- Daily ('+mobility_names[mobility_case]+')')
    plt.legend()
    plt.ylim(0,60000)
    plt.savefig(sub_dir+'/daily_Karnataka_'+label_names[parameter_case]+'_'+mobility_case+'.png', bbox_inches='tight')    
    # plt.show()
    plt.close()
    
    plt.plot(np.cumsum(karnataka_curve1_case1), 'b',label='Baseline')
    plt.plot(np.cumsum(karnataka_curve2_case1), 'g',label=label_names[parameter_case] + ' + 10%')
    plt.plot(np.cumsum(karnataka_curve3_case1), 'c',label=label_names[parameter_case] + ' - 10%')
    plt.plot(np.cumsum(karnataka_data), 'r',label='Reported cases')
    plt.xticks(x_indices, x_labels, rotation='vertical')
    plt.grid(True)
    plt.ylabel('Cumulative cases')
    plt.title('Karnataka -- Cumulative ('+mobility_names[mobility_case]+')') 
    plt.legend()
    plt.ylim(0,5000000)
    plt.savefig(sub_dir+'/cumulative_Karnataka'+label_names[parameter_case]+'_'+mobility_case+'.png', bbox_inches='tight')    
    # plt.show()
    plt.close()

mobility_cases = ['1000', '667', '500', '333']
parameter_cases = ['beta', 'kappa', 'lambda']
closed_loop = 0


label_names = {'beta': 'ContactRates', 'kappa': 'Shape', 'lambda': 'Scale'}
mobility_names = {'1000': 'No NPI', '667': '1/3 NPI', '500': '1/2 NPI', '333': '2/3 NPI'}

for i in range(0,4):
    for j in range(0,3):
        mobility_case = mobility_cases[i]
        parameter_case = parameter_cases[j]
        
        
        if closed_loop==1:
            path1 = '../../vaccine_allocation/closed_loop/controls/'
            path2 = '../../vaccine_allocation_'+parameter_case+'+10/closed_loop/controls/167000/'
            path3 = '../../vaccine_allocation_'+parameter_case+'-10/closed_loop/controls/167000/'
        else:
            path1 = '../../vaccine_allocation/fullrun_'+mobility_case+'/167000/'
            path2 = '../../vaccine_allocation_'+parameter_case+'+10/fullrun_'+mobility_case+'/167000/'
            path3 = '../../vaccine_allocation_'+parameter_case+'-10/fullrun_'+mobility_case+'/167000/'
        
        
        output1_case1 = path1+'August2021PopulationProportion.csv'
        output2_case1 = path2+'August2021PopulationProportion.csv'
        output3_case1 = path3+'August2021PopulationProportion.csv'
        
        if not (os.path.isdir('infected')):
            os.mkdir('infected')
        
        generate_plots_comparison(output1_case1,output2_case1,output3_case1, 204, 509, 509, 7, 'infected', 20)
