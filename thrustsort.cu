#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#define SIZE 16 << 20

  //compile with:
  // nvcc -m64 -arch=sm_35 thrustsort.cu -lcudart -O3 -o thrustsort
 // nvcc  thrustsort.cu -o thrustsort

int main(void)
{
    const uint    N = SIZE;
    cudaSetDevice (0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // generate 16M random numbers serially
    thrust::host_vector<int> h_vec_key(SIZE);
    thrust::host_vector<float> h_vec_value(SIZE);

    std::generate(h_vec_key.begin(), h_vec_key.end(), rand);
    std::generate(h_vec_value.begin(), h_vec_value.end(), rand);

    // double* h_arr = &h_vec_key[0];

    // transfer data to the device
    thrust::device_vector<int> d_vec_key = h_vec_key;
    thrust::device_vector<int> d_vec_value = h_vec_value;

    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    
    // sort data on the device (846M keys per second on GeForce GTX 480)
    // thrust::sort(d_vec_key.begin(), d_vec_key.end());
    thrust::sort_by_key(d_vec_key.begin(), d_vec_key.end(), d_vec_value.begin());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // transfer data back to host
    thrust::copy(d_vec_key.begin(), d_vec_key.end(), h_vec_key.begin());
    thrust::copy(d_vec_value.begin(), d_vec_value.end(), h_vec_value.begin());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double dTimeSecs = 1.0e-3 * elapsedTime ;
    printf("sortingNetworks-thrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u\n",
    (1.0e-6 * (double)N/dTimeSecs), dTimeSecs , N, 1);

    printf("Processing time: %f (ms)\n", elapsedTime);

    return 0;
}