# vaccine_allocation
## Steps to run simulations
* Clone the repository.
* Extract the travel matrices in fullrun_[NPI]/AgeDistrictMatrix.zip where NPI belongs to 1000, 667, 500 and 333.
* To generate outputs with 66% vaccine efifcacy, run fullrun_[NPI]/fullrun.py.
* To generate outputs with 33% vaccine efficacy, change 'VaxEfficacy=0.33' in configuration files (fullrun_[NPI]/AgeDistrict[AllocationStrategy]Configuration'). Then run fullrun_[NPI]/fullrun.py.

## Steps to run closed loop control
* Clone the repository.
* To run the closed loop control with 66% vaccine efficacy:
  * Extract closed_loop/May01_run/AgeDistrictMatrix.zip and run closed_loop/May01_run/100000/run.py
  * Change directory to closed_loop/controls/
  * Extract Output_0.zip and copy this to all sub-directories.
  * Run reset_configs.py, run.py, and merge_outputs.py in this order.
* To run the closed loop control with 33% vaccine efficacy:
  * Extract closed_loop/May01_run_33_vaxefficacy/AgeDistrictMatrix.zip and run closed_loop/May01_run_33_vaxefficacy/100000/run.py
  * Change directory to closed_loop/controls_33_vaxefficacy/
  * Extract Output_0.zip and copy this to all sub-directories
  * Run reset_configs.py, run.py, and merge_outputs.py in this order.

## Steps to generate plots
* Change the directory to get_results/plots.
* run generate_plots.sh.
