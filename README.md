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
For Validation, go to the Validation/ directory and replace the program inside the "Run the program or Kernel" in the validate.cu code (The program should be included in the file). Also the "just for test remove these for actual implementation" section should be commented out, because it is basically cheating as it copies to h_reference_keys and h_reference_values to h_keys and h_values. I was using it to test the validation program. Now compilation is possible using the following command:
  
  $ nvcc validate.cu -Icub-1.8.0/ -o validate ; ./validate "input items" "verbose" "device-id"

for example to create 20 random keys and validate with verbose option and with one device we can use the following code:

  $nvcc validate.cu -Icub-1.8.0/ -o validate ; ./validate 20 1 0
