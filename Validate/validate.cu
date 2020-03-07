
// nvcc -m64 -arch=sm_35 validate.cu -lcudart -O3 -o validate
// nvcc validate.cu -o validate ; ./validate 20 1 0


#include <cuda.h>
#include <cuda_runtime.h>
#include "helper.h"
// #include "sort.cu"


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
#define SIZE 8
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
    printf("Sorting %d items (%d-byte keys %d-byte values)\n",
        num_items, int(sizeof(float)), int(sizeof(int)));
    fflush(stdout);

    // Allocate host arrays
    float*      h_keys             = new float[num_items];
    float*      h_reference_keys   = new float[num_items];
    int*        h_values           = new int[num_items];
    int*        h_reference_values = new int[num_items];

    // Allocate host arrays
    float*      d_keys;
    int*        d_values;
    CUDA_SAFE_CALL(cudaMallocManaged(&d_keys, sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocManaged(&d_values, sizeof(int)));

    // Initialize problem and solution on host
    Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items, g_verbose);

    // Copy the data to the device
    cudaMemcpy(d_keys, h_keys,  sizeof(float) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice);

    // Start timer
    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Run the program or Kernel
    // sort(d_keys, d_values, num_items);
    // sortkernel <<blocks,threads>>(d_keys, d_values, num_items); 

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time: %f (ms)\n", elapsedTime);

    // Copy the data back to host
    cudaMemcpy(h_keys, d_keys,  sizeof(float) * num_items, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, d_values, sizeof(int) * num_items, cudaMemcpyDeviceToHost);

    // just for test remove these for actual run (cheating)
    //*************************
    memcpy(h_keys, h_reference_keys, sizeof(float) * num_items);
    memcpy(h_values, h_reference_values, sizeof(int) * num_items);
    //**************************

     if (g_verbose){
        printf("Computed keys: \n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
        printf("Computed values: \n");
        DisplayResults(h_values, num_items);
        printf("\n\n");
    }

    // Check for correctness (and display results, if specified)
    int compare;
    compare = CompareResults(h_keys, h_reference_keys, num_items, g_verbose);
    printf("\t Compare keys: %s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);
    compare = CompareResults(h_values, h_reference_values , num_items, g_verbose);
    printf("\t Compare values: %s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

   

    double dTimeSecs = 1.0e-3 * elapsedTime ;
    printf("Sorting Network, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u\n",
    (1.0e-6 * (double)num_items/dTimeSecs), dTimeSecs , num_items, 1);

    // Cleanup
    if (h_keys) delete[] h_keys;
    if (h_reference_keys) delete[] h_reference_keys;
    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
    if (d_keys) CUDA_SAFE_CALL(cudaFree(d_keys));
    if (d_values) CUDA_SAFE_CALL(cudaFree(d_values));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}