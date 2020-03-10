
// nvcc -m64 -arch=sm_35 validate_banyan.cu -lcudart -O3 -o validate_banyan
// nvcc validate_banyan.cu -o validate_banyan ; ./validate_banyan 16 1 0

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "helper_nov.h"
#include "banyan.cu"
// #include "../shared_memory/benes.cu"
using namespace std;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
#define SIZE 16
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
    printf("banyan_batcher in function call: N=%d - n=%d (sorting %d-byte keys) \n",(int)N,(int)n, int(sizeof(float)));

    fflush(stdout);

    // Allocate host arrays
    float*      h_data             = new float[N];
    float*      h_reference_data   = new float[N];

    // Allocate device arrays
    // copied from banyan.cu
    float*       d_data;
    CUDA_SAFE_CALL(cudaMallocManaged(&d_data, N * sizeof(float)));

    // Initialize problem and solution on host
    Initialize(h_data, h_reference_data, N, g_verbose);

    float hard_code[N]= {2,13,4,0,11,-5,9,1,15,-6,12,7,14,3,8,10};
    memcpy(h_data,hard_code,sizeof(float) * N);

    if (g_verbose){
        printf("Input keys: \n");
        DisplayResults(h_data, N);
        printf("\n\n");
    }

    // Copy the data to the device
    cudaMemcpy(d_data, h_data,  sizeof(float) * N, cudaMemcpyHostToDevice);

    // Start timer
    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Run the program or Kernel
    banyan(d_data , N, n);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time: %f (ms)\n", elapsedTime);

    // Copy the data back to host
    cudaMemcpy(h_data, d_data,  sizeof(float) * N, cudaMemcpyDeviceToHost);

    // just for test remove these for actual run (cheating)
    //*************************
    // memcpy(h_data, h_reference_data, sizeof(float) * N);
    //**************************


     if (g_verbose){
        printf("Computed keys: \n");
        DisplayResults(h_data, N);
        printf("\n\n");
    }

    // Check for correctness (and display results, if specified)
    int compare;
    compare = CompareResults(h_data, h_reference_data, N, g_verbose);
    printf("\t Compare keys: %s\n", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

   

    double dTimeSecs = 1.0e-3 * elapsedTime ;
    printf("Sorting Network, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u\n",
    (1.0e-6 * (double)N/dTimeSecs), dTimeSecs , N, 1);

    // Cleanup
    if (h_data) delete[] h_data;
    if (h_reference_data) delete[] h_reference_data;
    if (d_data) CUDA_SAFE_CALL(cudaFree(d_data));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}