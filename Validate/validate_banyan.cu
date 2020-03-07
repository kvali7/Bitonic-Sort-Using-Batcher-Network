
// nvcc -m64 -arch=sm_35 validate_banyan.cu -lcudart -O3 -o validate_banyan
// nvcc validate_banyan.cu -o validate_banyan ; ./validate_banyan 16 1 0

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "helper_nov.h"
#include "banyan.cu"
using namespace std;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
#define SIZE 8
bool    g_verbose = false;  // Whether to display input/output to console
ulong     num_items = SIZE;
int     deviceid = 0;

// MAIN
int main (int argc, char** argv){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    argsHandler (argc, argv, &num_items, &g_verbose, &deviceid);

    ulong N = num_items;  
    if (!IsPowerOfTwo(N)){
        fprintf(stderr, "Numberof items is not a power of two"
        "\n");
        exit(1);  
    }
    uint n = log2((double)N); // n is log2 of N
    cudaSetDevice (deviceid);

    // Discription
    printf("Sorting %d items (%d-byte keys) using Banyan Network, %d total stages\n",
        N, int(sizeof(float)), n);
    fflush(stdout);

    // Allocate host arrays
    float*      h_keys             = new float[N];
    float*      h_reference_keys   = new float[N];

    // Allocate device arrays
    // copied from benyan.cu
    float*       x;
    float*       y;
    bool*        comparators; 
    CUDA_SAFE_CALL(cudaMallocManaged(&x, N * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocManaged(&y, N * sizeof(float)));
    CUDA_SAFE_CALL(cudaMallocManaged(&comparators, N/2 * sizeof(bool)));

    // Initialize problem and solution on host
    Initialize(h_keys, h_reference_keys, N, g_verbose);

    // Copy the data to the device
    cudaMemcpy(x, h_keys,  sizeof(float) * N, cudaMemcpyHostToDevice);

    // Start timer
    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Run the program or Kernel
    benyan(x, y, comparators , N, n);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time: %f (ms)\n", elapsedTime);

    // Copy the data back to host
    cudaMemcpy(h_keys, x,  sizeof(float) * N, cudaMemcpyDeviceToHost);

    // just for test remove these for actual run (cheating)
    //*************************
    // memcpy(h_keys, h_reference_keys, sizeof(float) * N);
    //**************************


     if (g_verbose){
        printf("Computed keys: \n");
        DisplayResults(h_keys, N);
        printf("\n\n");
    }

    // Check for correctness (and display results, if specified)
    int compare;
    compare = CompareResults(h_keys, h_reference_keys, N, g_verbose);
    printf("\t Compare keys: %s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

   

    double dTimeSecs = 1.0e-3 * elapsedTime ;
    printf("Sorting Network, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u\n",
    (1.0e-6 * (double)N/dTimeSecs), dTimeSecs , N, 1);

    // Cleanup
    if (h_keys) delete[] h_keys;
    if (h_reference_keys) delete[] h_reference_keys;
    if (x) CUDA_SAFE_CALL(cudaFree(x));
    if (y) CUDA_SAFE_CALL(cudaFree(y));
    if (comparators) CUDA_SAFE_CALL(cudaFree(comparators));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}