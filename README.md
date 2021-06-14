# vaccine_allocation
## Steps to run simulations
* Clone the repository.
* ~~Extract the travel matrices in fullrun_[NPI]/AgeDistrictMatrix.zip where NPI belongs to 1000, 667, 500 and 333.~~
* To generate outputs with 66% vaccine efifcacy, run fullrun_[NPI]/fullrun.py.
* To generate outputs with 33% vaccine efficacy, change 'VaxEfficacy=0.33' in configuration files (fullrun_[NPI]/AgeDistrict[AllocationStrategy]Configuration). Then run fullrun_[NPI]/fullrun.py.

## Steps to run closed loop control
* Clone the repository.
* To run the closed loop control with 66% vaccine efficacy:
  *  ~~Extract closed_loop/May01_run/AgeDistrictMatrix.zip and~~ run closed_loop/May01_run/100000/run.py
  * Run closed_loop/copy_initial_results.py
  * Change directory to closed_loop/controls/
  * Run reset_configs.py, run.py, and merge_outputs.py in this order.
* To run the closed loop control with 33% vaccine efficacy, change 'VaxEfficacy=0.33' in the configuration file AgeDistrictConfigurationPopulationProportional under May01_run/100000/ and controls/[budget] where budget belongs to 100000, 133000, ... , 400000, and repeat the above steps.

## Steps to generate plots
* Change the directory to get_results/plots.
* run generate_plots.sh. 
* If there's a permission error, please run the following command: sudo chmod +x generate_plots.sh and enter the password. This will give the file executable permission.

### Reference
A. Adiga et al., Strategies to Mitigate COVID-19 Resurgence Assuming Immunity Waning: A Study for Karnataka, India. [medRxiv preprint](https://www.medrxiv.org/content/10.1101/2021.05.26.21257836v1)
