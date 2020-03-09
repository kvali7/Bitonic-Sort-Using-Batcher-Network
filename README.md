# CRS_PRJ_289Q_WQ2020_Temp
Branch for the Other GPU sorting algorithms

# CUB Radix Sort
For CUB radix sort, change the defined SIZE constant inside the cubradixsort.cu and compile with the following command
  
  $ nvcc  cubradixsort.cu -Icub-1.8.0/ -o cubradixsort

# Thrust Radix Sort
For Thrust radix sort, change the defined SIZE constant inside the thrustsort.cu and compile with the following command
  
  $ nvcc  thrustsort.cu -o thrustsort

# Bitonic Sort
For Bitonic sort, Go to the bitonicsort/ directory and change the defined SIZE constant inside the bitonicsortmain.cu and compile with the following command
  
  $ nvcc  bitonicsortmain.cu -Iinc -o bitonicsortmain


# Validation of our program
For Validation, go to the Validation/ directory and use with the following command

$ nvcc validate_banyan.cu -o validate_banyan ; ./validate_banyan 128

or for verbose use 

$ nvcc validate_banyan.cu -o validate_banyan ; ./validate_banyan 128 1
  
