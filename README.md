
# A CUDA-Based Banyan-Batcher Network for Bitonic Sort 
This repository contains code implementing a Banyan-Batcher sorting network for the final project of the Winter 2020 offering of EEC289Q Modern Parallel Computing at UC Davis.

## Presentation
[![Link to final presentation](https://img.youtube.com/vi/At_atdszObM/0.jpg)](https://www.youtube.com/watch?v=At_atdszObM)

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
  
