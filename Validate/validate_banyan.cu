// nvcc -m64 -arch=sm_35 validate_banyan.cu -lcudart -O3 -o validate_banyan
// nvcc validate_banyan.cu -o validate_banyan ; ./validate_banyan 16 10 1 0

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "helper_nov.h"
#include "banyan.cu"
using namespace std;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
#define SIZE 16
bool    g_verbose = false;  // Whether to display input/output to console
ulong     num_items = SIZE;
int     deviceid = 0;
int     thresh  = 4;

// MAIN
int main (int argc, char** argv){

    double time_taken = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // argsHandler (argc, argv, &num_items, &g_verbose, &deviceid);
    argsHandlerThresh (argc, argv, &num_items, &g_verbose, &deviceid, &thresh);

    ulong N = num_items;  
    if (thresh > N/2)
        thresh = N/2 + 1;

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


    // Initialize problem and solution on host Random
    // Initialize(h_data, h_reference_data, N, g_verbose);
    // Initialize(h_data, h_reference_data, N, g_verbose, &time_taken);
    // std::cout << "Time taken by std::sort on CPU is : " << fixed 
    // << time_taken * 1.0e3 << setprecision(9); 
    // std::cout << " msec" << " \t and " ; 
    // std::cout << "Speed by program on CPU is : " << fixed 
    //     << 1.0e-6 * (double)num_items/time_taken << setprecision(5); 
    // std::cout << " MElements/s" << endl; 
    // Initialize problem and solution on host with hardcoded array
    // float hard_code[] = {2,13,4,0,11,-5,9,1,15,-6,12,7,14,3,8,10};
    // memcpy(h_data,hard_code,sizeof(float) * N);
    // float hard_code_sortd[] = {-6, -5, 0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    // memcpy(h_reference_data,hard_code_sortd,sizeof(float) * N);

    // Initialize problem and solution with descending array
    for (int k = 0; k < N; k++){
        h_reference_data[k] = (float)k;
        h_data[k] = (float)N-k-1;
    }
    if (g_verbose){
        printf("Input keys: \n");
        DisplayResultsHT(h_data, N, thresh);
        // DisplayResults(h_data, N);
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
        printf("Reference keys: \n");
        // DisplayResults(h_reference_data, N);
        DisplayResultsHT(h_reference_data, N, thresh);
        printf("\n\n");
        printf("Computed keys: \n");
        // DisplayResults(h_data, N);
        DisplayResultsHT(h_data, N, thresh);
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