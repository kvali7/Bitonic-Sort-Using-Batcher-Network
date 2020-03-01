#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

  //compile with:
  // nvcc -m64 -arch=sm_35 thrustsort.cu -lcudart -O3 -o thrustsort
 // nvcc  thrustsort.cu -o thrustsort

int main(void)
{
    cudaSetDevice (0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // generate 32M random numbers serially
    thrust::host_vector<int> h_vec(16 << 20);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    printf("Processing time: %f (ms)\n", elapsedTime);

    return 0;
}