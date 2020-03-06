
// nvcc -m64 -arch=sm_35 validate_nov.cu -lcudart -O3 -o validate_nov
// nvcc validate_nov.cu -o validate_nov ; ./validate_nov 20 1 0


#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_nov.h"
// #include "sort.cu"


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
#define SIZE 6
bool    g_verbose = false;  // Whether to display input/output to console
int     num_items = SIZE;
int     deviceid = 0;

// MAIN
int main (int argc, char** argv){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    argsHandler (argc, argv, &num_items, &g_verbose, &deviceid);

    cudaSetDevice (deviceid);

    // Discription
    printf("Sorting %d items (%d-byte keys)\n",
        num_items, int(sizeof(float)), int(sizeof(int)));
    fflush(stdout);

    // Allocate host arrays
    float*      h_keys             = new float[num_items];
    float*      h_reference_keys   = new float[num_items];

    // Allocate host arrays
    float*      d_keys;
    CUDA_SAFE_CALL(cudaMallocManaged(&d_keys, sizeof(float)));

    // Initialize problem and solution on host
    Initialize(h_keys, h_reference_keys, num_items, g_verbose);

    // Copy the data to the device
    cudaMemcpy(d_keys, h_keys,  sizeof(float) * num_items, cudaMemcpyHostToDevice);

    // Start timer
    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Run the program or Kernel
    // sort(d_keys, num_items);
    // sortkernel <<blocks,threads>>(d_keys, num_items); 

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time: %f (ms)\n", elapsedTime);

    // Copy the data back to host
    cudaMemcpy(h_keys, d_keys,  sizeof(float) * num_items, cudaMemcpyDeviceToHost);

    // just for test remove these for actual implementation
    //*************************
    memcpy(h_keys, h_reference_keys, sizeof(float) * num_items);
    //**************************
    

    // Check for correctness (and display results, if specified)
    int compare;
    compare = CompareResults(h_keys, h_reference_keys, num_items, g_verbose);
    printf("\t Compare keys: %s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    if (g_verbose){
        printf("Computed keys: \n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
    }

    double dTimeSecs = 1.0e-3 * elapsedTime ;
    printf("Sorting Network, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u\n",
    (1.0e-6 * (double)num_items/dTimeSecs), dTimeSecs , num_items, 1);

    // Cleanup
    if (h_keys) delete[] h_keys;
    if (h_reference_keys) delete[] h_reference_keys;
    if (d_keys) CUDA_SAFE_CALL(cudaFree(d_keys));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}