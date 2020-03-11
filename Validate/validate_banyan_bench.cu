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
#define SIZE 16
bool    g_verbose = false;  // Whether to display input/output to console
ulong     num_items = SIZE;
int     deviceid = 0;
ulong N;

// MAIN
int main (int argc, char** argv){

    cudaSetDevice (deviceid);

    int minn = 4;
    int maxn = 26;
    int maxiter = 8;
    double time[30] = {0};
    double speed[30] = {0};
    double timetemp[10] = {0};
    double speedtemp[10] = {0};
    double timesum = 0;
    double speedsum = 0;

    int n = 24;
    for ( n = minn; n <= maxn; n++){
      N = pow(2,n);
      memset (timetemp, 0, 10* sizeof(double));
      memset (speedtemp, 0 , 10* sizeof(double));
      for (int iter = 0; iter < maxiter; iter++){

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // argsHandler (argc, argv, &num_items, &g_verbose, &deviceid);

            // ulong N = num_items;  
            if (!IsPowerOfTwo(N)){
                fprintf(stderr, "Numberof items is not a power of two"
                "\n");
                exit(1);  
            }
            // uint n = log2((double)N); // n is log2 of N

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


            timetemp[iter] = elapsedTime;
            speedtemp[iter] = 1.0e-6 * (double)N/dTimeSecs;
        }
    
        timesum = 0;
        speedsum = 0;
        for (int iter = 0; iter < maxiter; iter++){
          timesum = timesum + timetemp[iter];
          speedsum = speedsum + speedtemp[iter];
        }
    
        time[n] = (float)timesum / maxiter;
        speed[n] = (float)speedsum / maxiter;
      }
    
    
    
    
      printf ("\n\n\tn:\n");
      for (n = minn; n <= maxn; n++){
        printf ("%d,\t", n);
      }
    
      printf ("\n\n\tN:\n");
      for (n = minn; n <= maxn; n++){
        printf ("%u,\t", (uint)pow(2,n));
      }
      
      printf ("\n\n\ttime:\n");
      for (n = minn; n <= maxn; n++){
        printf ("%lf,\t", time[n]);
      }
    
      printf ("\n\n");
    
      printf ("\n\n\tspeed:\n");
      for (n = minn; n <= maxn; n++){
        printf ("%lf,\t", speed[n]);
      }    
    
      printf ("\n\n");
    
    
}