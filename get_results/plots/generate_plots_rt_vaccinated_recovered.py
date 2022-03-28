import sys
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def get_districts():
    return pd.read_csv('../../seroprevalence_modified.csv')['Unit'].tolist()


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
    inf = pd.read_csv('../../target_curves/tinf_df.csv')
    
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
    sero_data = pd.read_csv('../../seroprevalence_modified.csv')
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
    sero_data = pd.read_csv('../../seroprevalence_modified.csv')
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



def prepare_district_simulation_plots_susceptible(outputfile, days_plot):
    jump_values = 7
    
    districts = get_districts()
    simulation = pd.read_csv(outputfile, header=None, sep=' ')
    district_simulation = pd.DataFrame(columns=['DistrictName']+(list(range(0,days_plot))))
    district_simulation['DistrictName'] = districts
    sero_data = pd.read_csv('../../seroprevalence_modified.csv')
    for x in districts:
        for i in range(days_plot):
            district_simulation.at[district_simulation['DistrictName']==x, i] = \
                (np.sum(simulation.loc[simulation[0].str.contains(x), i+1].tolist())).astype(int)
    return district_simulation



def get_cir_factors(duration):
    testing_data = pd.read_excel('../../target_curves/testing.xlsx')
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


def get_population():
    data = pd.read_csv('../../AgeDistrictPopulation.csv', sep = ' ', header=None)
    df = pd.DataFrame(columns = ['District', 'pop'])
    districts = get_districts()
    for x in districts:
        population = np.sum(data.loc[data[0].str.contains(x),1].values.tolist())
        new_row  = {'District':x, 'pop':population}
        df = df.append(new_row, ignore_index=True)
    return df


def generate_plots_rt(output1_case1,output2_case1,output3_case1,output4_case1, output5_case1, output1_case2,output2_case2,output3_case2,output4_case2, output5_case2, betas_file,start_index, plot_days_data, plot_days_simulation, smooth_days, sub_dir, start_offset, recovery_rate):
    delay = 4
    if start_offset == 20:
        x_indices_temp = [186, 202, 233, 263, 294, 325, 355, 386, 416, 447, 478, 505]
        x_indices = [a-(start_offset+1) - x_indices_temp[0]
                     for a in x_indices_temp][1::]
        x_labels = [ '01 May 2021', '01 Jun 2021', '01 Jul 2021', '01 Aug 2021', '01 Sep 2021', '01 Oct 2021', '01 Nov 2021', '01 Dec 2021', '01 Jan 2022', '01 Feb 2022', '28 Feb 2022']
    else:
        x_indices = [0, 21, 51, 82, 113, 141, 172, 202, 233, 263, 294, 324]
        x_labels = ['11 Oct 2020', '01 Nov 2020', '01 Dec 2020', '01 Jan 2021', '01 Feb 2021', '01 Mar 2021', '01 Apr 2021', '01 May 2021', '01 Jun 2021', '01 Jul 2021', '01 Aug 2021', '31 Aug 2021']
    district_simulation_1_case1 = pd.read_csv(output1_case1, header=None, sep = ' ')
    district_simulation_2_case1 = pd.read_csv(output2_case1, header=None, sep = ' ')
    district_simulation_3_case1 = pd.read_csv(output3_case1, header=None, sep = ' ')
    district_simulation_4_case1 = pd.read_csv(output4_case1, header=None, sep = ' ')
    district_simulation_5_case1 = pd.read_csv(output5_case1, header=None, sep = ' ')


    district_simulation_1_case2 = pd.read_csv(output1_case2, header=None, sep = ' ')
    district_simulation_2_case2 = pd.read_csv(output2_case2, header=None, sep = ' ')
    district_simulation_3_case2 = pd.read_csv(output3_case2, header=None, sep = ' ')
    district_simulation_4_case2 = pd.read_csv(output4_case2, header=None, sep = ' ')
    district_simulation_5_case2 = pd.read_csv(output5_case2, header=None, sep = ' ')

    
    betas = pd.read_csv(betas_file, header=None, sep = ' ')
    
    district_simulation_1_case1 = district_simulation_1_case1.set_index(0)
    district_simulation_2_case1 = district_simulation_2_case1.set_index(0)
    district_simulation_3_case1 = district_simulation_3_case1.set_index(0)
    district_simulation_4_case1 = district_simulation_4_case1.set_index(0)
    district_simulation_5_case1 = district_simulation_5_case1.set_index(0)

    district_simulation_1_case2 = district_simulation_1_case2.set_index(0)
    district_simulation_2_case2 = district_simulation_2_case2.set_index(0)
    district_simulation_3_case2 = district_simulation_3_case2.set_index(0)
    district_simulation_4_case2 = district_simulation_4_case2.set_index(0)
    district_simulation_5_case2 = district_simulation_5_case2.set_index(0)
    
    district_simulation_1_case1.sort_index(inplace = True)
    district_simulation_2_case1.sort_index(inplace = True)
    district_simulation_3_case1.sort_index(inplace = True)
    district_simulation_4_case1.sort_index(inplace = True)
    district_simulation_5_case1.sort_index(inplace = True)
    
    district_simulation_1_case2.sort_index(inplace = True)
    district_simulation_2_case2.sort_index(inplace = True)
    district_simulation_3_case2.sort_index(inplace = True)
    district_simulation_4_case2.sort_index(inplace = True)
    district_simulation_5_case2.sort_index(inplace = True)


    betas = betas.set_index(0)
    betas.sort_index(inplace = True)
    age_districts = betas.index.values
    
    district_simulation_5_case1 = district_simulation_5_case1[list(range(1,plot_days_simulation+1))]
    district_simulation_5_case2 = district_simulation_5_case2[list(range(1,plot_days_simulation+1))]

    curve1_case1_df = pd.DataFrame(betas.values*district_simulation_1_case1.values, columns = betas.columns, index = betas.index)
    curve2_case1_df = pd.DataFrame(betas.values*district_simulation_2_case1.values, columns = betas.columns, index = betas.index)
    curve3_case1_df = pd.DataFrame(betas.values*district_simulation_3_case1.values, columns = betas.columns, index = betas.index)
    curve4_case1_df = pd.DataFrame(betas.values*district_simulation_4_case1.values, columns = betas.columns, index = betas.index)
    curve5_case1_df = pd.DataFrame(betas.values*district_simulation_5_case1.values, columns = betas.columns, index = betas.index)
    
    curve1_case2_df = district_simulation_1_case2
    curve2_case2_df = district_simulation_2_case2
    curve3_case2_df = district_simulation_3_case2
    curve4_case2_df = district_simulation_4_case2
    curve5_case2_df = district_simulation_5_case2
    
    M = pd.read_csv('../../target_curves/M.csv', header=None)
    M = np.transpose(M)
    age_district_population_df = pd.read_csv('../../AgeDistrictPopulation.csv', sep = ' ', header=None)
    age_district_population_df = age_district_population_df.set_index(0)
    age_district_population_df.sort_index(inplace = True)
    age_district_population = age_district_population_df[1].values

    districts = [age_districts[i].split('++')[0] for i in np.arange(0,266,7)]
    
    def get_inside_den_curve(curve1_case2_df, curve1_case1_df):
        
        inside_num_curve1_df_0 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[0].values, 38), axis = 0))
        inside_num_curve1_df_1 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[1].values, 38), axis = 0))
        inside_num_curve1_df_2 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[2].values, 38), axis = 0))
        inside_num_curve1_df_3 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[3].values, 38), axis = 0))
        inside_num_curve1_df_4 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[4].values, 38), axis = 0))
        inside_num_curve1_df_5 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[5].values, 38), axis = 0))
        inside_num_curve1_df_6 = curve1_case2_df.multiply(curve1_case1_df.multiply(np.tile(M.loc[6].values, 38), axis = 0))
        inside_num_curve1_df = pd.DataFrame(columns = ['age_districts']+list(range(1,plot_days_simulation+1)))
        
        
        inside_num_curve1_df['age_districts'] = age_districts
        inside_num_curve1_df = inside_num_curve1_df.set_index('age_districts')
        
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+0] =  np.sum(inside_num_curve1_df_0.iloc[[j*7+k for k in range(0,7)]], axis = 0)
            
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+1] =  np.sum(inside_num_curve1_df_1.iloc[[j*7+k for k in range(0,7)]], axis = 0)
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+2] =  np.sum(inside_num_curve1_df_2.iloc[[j*7+k for k in range(0,7)]], axis = 0)
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+3] =  np.sum(inside_num_curve1_df_3.iloc[[j*7+k for k in range(0,7)]], axis = 0)
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+4] =  np.sum(inside_num_curve1_df_4.iloc[[j*7+k for k in range(0,7)]], axis = 0)
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+5] =  np.sum(inside_num_curve1_df_5.iloc[[j*7+k for k in range(0,7)]], axis = 0)
        for j in range(0,38):
            inside_num_curve1_df.iloc[j*7+6] =  np.sum(inside_num_curve1_df_6.iloc[[j*7+k for k in range(0,7)]], axis = 0)
        
        return inside_num_curve1_df
    
    Mprime = pd.DataFrame(np.ones((7,7)))
    def get_common_denom():
        temp = []
        for j in range(0,38):
            temp = temp + [np.sum((np.tile(Mprime.loc[k].values, 38)*age_district_population)[j*7+0:j*7+6]) for k in range(0, 7)]
        return np.array(temp)
    
    def get_common_denom_df():
        return age_district_population
    
    for i in range(len(districts)):
        denom_curve1 = recovery_rate*(np.sum(district_simulation_1_case2.iloc[i*7:i*7+7].values, axis = 0)) #denom_curve1 = recovery_rate*(district_simulation_1_case2.sum().values)
        denom_curve2 = recovery_rate*(np.sum(district_simulation_2_case2.iloc[i*7:i*7+7].values, axis = 0)) #denom_curve2 = recovery_rate*(district_simulation_2_case2.sum().values)
        denom_curve3 = recovery_rate*(np.sum(district_simulation_3_case2.iloc[i*7:i*7+7].values, axis = 0)) #denom_curve3 = recovery_rate*(district_simulation_3_case2.sum().values)
        denom_curve4 = recovery_rate*(np.sum(district_simulation_4_case2.iloc[i*7:i*7+7].values, axis = 0)) #denom_curve4 = recovery_rate*(district_simulation_4_case2.sum().values)
        denom_curve5 = recovery_rate*(np.sum(district_simulation_5_case2.iloc[i*7:i*7+7].values, axis = 0)) #denom_curve5 = recovery_rate*(district_simulation_5_case2.sum().values)
        
        
        inside_den_curve_common = get_common_denom_df()
    
        inside_num_curve1_df = get_inside_den_curve(curve1_case2_df, curve1_case1_df)
        inside_num_curve2_df = get_inside_den_curve(curve2_case2_df, curve2_case1_df)
        inside_num_curve3_df = get_inside_den_curve(curve3_case2_df, curve3_case1_df)
        inside_num_curve4_df = get_inside_den_curve(curve4_case2_df, curve4_case1_df)
        inside_num_curve5_df = get_inside_den_curve(curve5_case2_df, curve5_case1_df)
        
        num_curve1 = np.sum((inside_num_curve1_df.divide(inside_den_curve_common, axis = 0)).iloc[i*7:i*7+7].values, axis = 0) #num_curve1 = (inside_num_curve1_df.divide(inside_den_curve_common, axis = 0)).sum().values
        num_curve2 = np.sum((inside_num_curve2_df.divide(inside_den_curve_common, axis = 0)).iloc[i*7:i*7+7].values, axis = 0)#num_curve2 = (inside_num_curve2_df.divide(inside_den_curve_common, axis = 0)).sum().values
        num_curve3 = np.sum((inside_num_curve3_df.divide(inside_den_curve_common, axis = 0)).iloc[i*7:i*7+7].values, axis = 0)#num_curve3 = (inside_num_curve3_df.divide(inside_den_curve_common, axis = 0)).sum().values
        num_curve4 = np.sum((inside_num_curve4_df.divide(inside_den_curve_common, axis = 0)).iloc[i*7:i*7+7].values, axis = 0)#num_curve4 = (inside_num_curve4_df.divide(inside_den_curve_common, axis = 0)).sum().values
        num_curve5 = np.sum((inside_num_curve5_df.divide(inside_den_curve_common, axis = 0)).iloc[i*7:i*7+7].values, axis = 0)#num_curve5 = (inside_num_curve5_df.divide(inside_den_curve_common, axis = 0)).sum().values
        
        curve1 = ((num_curve1/denom_curve1)[start_offset+delay::])[186::]
        curve2 = ((num_curve2/denom_curve2)[start_offset+delay::])[186::]
        curve3 = ((num_curve3/denom_curve3)[start_offset+delay::])[186::]
        curve4 = ((num_curve4/denom_curve4)[start_offset+delay::])[186::]
        curve5 = ((num_curve5/denom_curve5)[start_offset+delay::])[186::]
        
        
        plt.rcParams['figure.figsize'] = [18, 10]
        plt.rcParams['font.size'] = 26
        
        plt.plot((curve1*1), 'b',label='No NPI')
        plt.plot((curve2*(1/3)), 'g',label='1/3 NPI')
        plt.plot((curve3*(1/2)), 'c',label='1/2 NPI')
    
        plt.plot((curve4*(2/3)), 'm',label='2/3 NPI')
        
        #plt.plot((curve5), 'k',label='Closed Loop Control')
        plt.xticks(x_indices, x_labels, rotation='vertical')
        plt.grid(True)
        plt.ylabel('R_t')
        plt.title(districts[i]+' -- Basic reproduction number (R_t)') 
        plt.legend()
        plt.savefig(sub_dir+'/daily_'+districts[i]+'.png', bbox_inches='tight')    
        # plt.show()
        plt.close()
    

def generate_plots_recovered_fraction(output1_case1,output2_case1,output3_case1,output4_case1, output5_case1,start_index, plot_days_data, plot_days_simulation, smooth_days, sub_dir, start_offset):
    delay = 4
    if start_offset == 20:
        x_indices_temp = [186, 202, 233, 263, 294, 325, 355, 386, 416, 447, 478, 505]
        x_indices = [a-(start_offset+1) - x_indices_temp[0]
                     for a in x_indices_temp][1::]
        x_labels = [ '01 May 2021', '01 Jun 2021', '01 Jul 2021', '01 Aug 2021', '01 Sep 2021', '01 Oct 2021', '01 Nov 2021', '01 Dec 2021', '01 Jan 2022', '01 Feb 2022', '28 Feb 2022']
    else:
        x_indices = [0, 21, 51, 82, 113, 141, 172, 202, 233, 263, 294, 324]
        x_labels = ['11 Oct 2020', '01 Nov 2020', '01 Dec 2020', '01 Jan 2021', '01 Feb 2021', '01 Mar 2021', '01 Apr 2021', '01 May 2021', '01 Jun 2021', '01 Jul 2021', '01 Aug 2021', '31 Aug 2021']
    curve1_case1_df = pd.read_csv(output1_case1, header=None, sep = ' ')
    curve2_case1_df = pd.read_csv(output2_case1, header=None, sep = ' ')
    curve3_case1_df = pd.read_csv(output3_case1, header=None, sep = ' ')
    curve4_case1_df = pd.read_csv(output4_case1, header=None, sep = ' ')
    curve5_case1_df = pd.read_csv(output5_case1, header=None, sep = ' ')
    
    
    curve1_case1_df = curve1_case1_df.set_index(0)
    curve2_case1_df = curve2_case1_df.set_index(0)
    curve3_case1_df = curve3_case1_df.set_index(0)
    curve4_case1_df = curve4_case1_df.set_index(0)
    curve5_case1_df = curve5_case1_df.set_index(0)

    curve5_case1_df = curve5_case1_df[list(range(1,plot_days_simulation+1))]
    
    curve1_case1_df.sort_index(inplace = True)
    curve2_case1_df.sort_index(inplace = True)
    curve3_case1_df.sort_index(inplace = True)
    curve4_case1_df.sort_index(inplace = True)
    curve5_case1_df.sort_index(inplace = True)
    
    population_df = get_population()
    population = np.sum(population_df['pop'].values.tolist())
    
    age_district_population_df = pd.read_csv('../../AgeDistrictPopulation.csv', sep = ' ', header=None)
    age_district_population_df = age_district_population_df.set_index(0)
    age_district_population_df.sort_index(inplace = True)
    age_district_population = age_district_population_df[1].values
    age_districts = age_district_population_df.index.values
    
    districts = [age_districts[i].split('++')[0] for i in np.arange(0,266,7)]
    
    for i in range(len(districts)):
        population = np.sum(age_district_population[i*7:i*7+7])
    
        curve1_case1 = np.sum((curve1_case1_df.iloc[i*7:i*7+7].values), axis = 0)[start_offset+delay::][186::]
        curve2_case1 = np.sum((curve2_case1_df.iloc[i*7:i*7+7].values), axis = 0)[start_offset+delay::][186::]
        curve3_case1 = np.sum((curve3_case1_df.iloc[i*7:i*7+7].values), axis = 0)[start_offset+delay::][186::]
        curve4_case1 = np.sum((curve4_case1_df.iloc[i*7:i*7+7].values), axis = 0)[start_offset+delay::][186::]
        curve5_case1 = np.sum((curve5_case1_df.iloc[i*7:i*7+7].values), axis = 0)[start_offset+delay::][186::]
    
    
    
        plt.plot((curve1_case1/population), 'b',label='No NPI')
        plt.plot((curve2_case1/population), 'g',label='1/3 NPI')
        plt.plot((curve3_case1/population), 'c',label='1/2 NPI')
    
        plt.plot((curve4_case1/population), 'm',label='2/3 NPI')
        
        #plt.plot((curve5_case1/population), 'k',label='Closed Loop Control')
        plt.xticks(x_indices, x_labels, rotation='vertical')
        plt.grid(True)
        plt.ylabel('Recovered fraction')
        plt.title(districts[i]+' -- Recovered fraction') 
        plt.legend()
        plt.savefig(sub_dir+'/daily_'+districts[i]+'.png', bbox_inches='tight')    
        # plt.show()
        plt.close()

path1 = '../../fullrun_1000/167000/' 
path2 = '../../fullrun_667/167000/'
path3 = '../../fullrun_500/167000/'
path4 = '../../fullrun_333/167000/'
path5 = '../../closed_loop/controls/167000/'

output1_case1 = path1+'August2021PopulationProportion.csv_susceptible.csv'
output2_case1 = path2+'August2021PopulationProportion.csv_susceptible.csv'
output3_case1 = path3+'August2021PopulationProportion.csv_susceptible.csv'
output4_case1 = path4+'August2021PopulationProportion.csv_susceptible.csv'
output5_case1 = path5+'OutputMerged.csv_susceptible.csv'
output1_case2 = path1+'August2021PopulationProportion.csv_infected.csv'
output2_case2 = path2+'August2021PopulationProportion.csv_infected.csv'
output3_case2 = path3+'August2021PopulationProportion.csv_infected.csv'
output4_case2 = path4+'August2021PopulationProportion.csv_infected.csv'
output5_case2 = path5+'OutputMerged.csv_infected.csv'

betas_file = '../../AgeDistrictExposureRates.csv'
if not (os.path.isdir('r_t')):
    os.mkdir('r_t')

recovery_rate = 0.2
generate_plots_rt(output1_case1,output2_case1,output3_case1,output4_case1, output5_case1, output1_case2,output2_case2,output3_case2,output4_case2, output5_case2, betas_file, 204, 509, 509, 7, 'R_t', 20, recovery_rate)


output1_case1 = path1+'August2021PopulationProportion.csv_recovered.csv'
output2_case1 = path2+'August2021PopulationProportion.csv_recovered.csv'
output3_case1 = path3+'August2021PopulationProportion.csv_recovered.csv'
output4_case1 = path4+'August2021PopulationProportion.csv_recovered.csv'
output5_case1 = path5+'OutputMerged.csv_recovered.csv'
betas = '../../AgeDistrictExposureRates.csv'
if not (os.path.isdir('recovered_fraction')):
    os.mkdir('recovered_fraction')


generate_plots_recovered_fraction(output1_case1,output2_case1,output3_case1,output4_case1, output5_case1, 204, 509, 509, 7, 'recovered_fraction', 20)


