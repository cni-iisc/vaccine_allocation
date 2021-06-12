# vaccine_allocation
## Steps to run simulations
* Clone the repository
* Extract the travel matrices in fullrun_npi/AgeDistrictMatrix.zip where npi belongs to 1000, 667, 500 and 333.
* run fullrun_npi/fullrun.py to generate the outputs

## Steps to run closed loop control
* Clone the repository
* To run the closed loop control with 66% vaccine efficacy:
  * Extract closed_loop/May01_run/AgeDistrictMatrix.zip and run closed_loop/May01_run/100000/run.py
  * Change directory to closed_loop/controls/
  * Run reset_configs.py, run.py and merge_outputs.py in this order.
* To run the closed loop control with 33% vaccine efficacy:
  * Extract closed_loop/May01_run_33_vaxefficacy/AgeDistrictMatrix.zip and run closed_loop/May01_run_33_vaxefficacy/100000/run.py
  * Change directory to closed_loop/controls_33_vaxefficacy/
  * Run reset_configs.py, run.py and merge_outputs.py in this order.
