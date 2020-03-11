#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <thrust/swap.h>
using namespace std;

/* Cuda memcheck snippets from HW3 
 * http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
 */
#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaDeviceSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)

// HOST helper function: get N given size of list
int getN(int size) {
  int N = 2; 
  while (N < size) {
    N = N*2;
  }
  return N;
}

// SHUFFLE kernel for N=pow(2,n) elements
__global__
void shuffleN(float *in, float *out, int size, int N) {
  // TO-DO: pull in shared memory
  // unsigned long ind_in  = threadIdx.x; // thread within block
  // unsigned long ind_in_base = blockIdx.x*blockDim.x; // block within col
  unsigned long ind_in_base = blockIdx.x*size; // block within col; accommodate fixed max threadNum=1024
  unsigned long ind_out; 
  unsigned long stride = blockDim.x; 
  for (unsigned long ind_in = threadIdx.x; ind_in < (unsigned long) size; ind_in += stride) {
    // ind_out = 2*(ind_in-ind_in % size/2);
    if (ind_in % 2 == 0) { // even 
      ind_out = ind_in/2;
    }  
    else {  // odd 
      ind_out = (size/2)+(ind_in/2);
    }
    out[ind_in_base+ind_in]=in[ind_in_base+ind_out];
    // using "out" as ping-pong; write back to "in"
    // __syncthreads();
    // in[ind_in_base+ind_in]=out[ind_in_base+ind_in];
  }
  __syncthreads();
  for (unsigned long ind_in = threadIdx.x; ind_in < (unsigned long) size; ind_in += stride) {
    in[ind_in_base+ind_in]=out[ind_in_base+ind_in];
  }
}


// BUTTERFLY kernel for N=pow(2,n) elements
__global__
void butterflyN(float *in, float *out, int size, int N) {
  // TO-DO: pull in shared memory
  unsigned long ind_in_base = blockIdx.x*size; // block within col; accommodate fixed max threadNum=1024
  unsigned long ind_out; 
  unsigned long stride = blockDim.x; 
  for (unsigned long ind_in = threadIdx.x; ind_in < (unsigned long) size; ind_in += stride) {
    if (ind_in < size/2) { // first half of list
      if (ind_in%2==0) // even
        ind_out = ind_in;
      else // odd
        ind_out = ind_in + (size/2 - 1);
    }  
    else {  // second half of list
      if (ind_in%2==0) // even
        ind_out = ind_in - (size/2 - 1);
      else //odd
        ind_out = ind_in;
    }
    out[ind_in_base+ind_in]=in[ind_in_base+ind_out];
  }
  // using "out" as ping-pong; write back to "in"
  __syncthreads();
  for (unsigned long ind_in = threadIdx.x; ind_in < (unsigned long) size; ind_in += stride) {
    in[ind_in_base+ind_in]=out[ind_in_base+ind_in];
  }
}

// COMPARE AND SWAP 
__global__
void compareAndSwap(float *in, int level, int N) {
  // TO-DO: pull in shared memory
  unsigned long stride = blockDim.x*gridDim.x; 
  // if (comp_ind < N) {
  for (unsigned long comp_ind = (unsigned long) blockDim.x*blockIdx.x+threadIdx.x;
       comp_ind < N;
       comp_ind += stride) {
    // STEP 0: Get direction of comparison
    bool comp_bool = (bool)(comp_ind & (0x1 << level)); 
    // STEP 1: Write out results of comparisons
    unsigned long data0 = 2*comp_ind;
    unsigned long data1 = 2*comp_ind+1;
    // compare and swap based on comp_bool
    if ( (comp_bool && in[data0] < in[data1]) || (!comp_bool && in[data0] >= in[data1]) ) { 
        thrust::swap(*(in+data0), *(in+data1)); 
    }
  }
}

void banyan_batcher(float *x, int N, int thresh) {
  // bitonic mergesort on batcher-banyan network
  int n = (int) log2((float)N);
  printf("banyan_batcher in function call: N=%d - n=%d\n",N,n);
  float *y;
  int level = 0;
  int stage = 0;
  int substage = 0;
  int div; // blockNum for current routing kernel
  int threadNum; // threadNum for current routing kernel
  int compThreadNum = 512; // threadNum for compareAndSwap kernel
  int compBlockNum = min(65535,(N+compThreadNum-1)/compThreadNum); // threadNum for compareAndSwap kernel; max(blockNum)=65535
  CUDA_SAFE_CALL(cudaMallocManaged(&y, N*sizeof(float)));
  // printf("compareAndSwap blockNum=%d - threadNum=%d\n", compBlockNum, compThreadNum);
  while (stage < n) {
    while (substage <= stage) { 
      div = N/(pow(2,2+stage-substage));
      printf("stage=%d - substage=%d - div=%d - level=%d\n", stage, substage, div, level);
      if (stage < n-1) {
        threadNum = min(1024, N/div); 
        if (substage == 0) {
          compareAndSwap<<<compThreadNum,compBlockNum>>>(x, level, N);
          cudaDeviceSynchronize();
          printf("-> compare for stage=%d at level=%d\n", stage, level);
          shuffleN<<<div,threadNum>>>(x, y, N/div, N);
          cudaDeviceSynchronize();
          printf("-> shuffle for stage=%d\n", stage);
          level++;
        }
        compareAndSwap<<<compThreadNum,compBlockNum>>>(x, level, N);
        cudaDeviceSynchronize();
        printf("-> compare for stage=%d at level=%d\n", stage, level);
        butterflyN<<<div,threadNum>>>(x, y, N/div, N);
        cudaDeviceSynchronize();
        printf("-> butterfly for stage=%d, substage=%d\n", stage, substage);
        substage++;
      }
      else {
        compareAndSwap<<<compThreadNum,compBlockNum>>>(x, level, N);
        cudaDeviceSynchronize();
        printf("-> final compare for stage=%d at level=%d\n", stage, level);
        break;
      }
    }
    stage++;
    substage = 0;
  }
  CUDA_SAFE_CALL(cudaFree(y));
  printf("-> first and last elements of sorted result for N=%d\n", N);
  for (int i=0; i<2*thresh-1; i++)
    printf("for i=%d: x=%f\n", (i<thresh) ? i : N-(2*thresh-i-1) , (i<thresh) ? x[i] : x[N-(2*thresh-i-1)]);
}

int main(int argc, char** argv)
{
  // USAGE: single argument
  // -> n = argv[1]
  // --> e.g. "./banyan 4" would run n=4, N=16

  // params for testing helper functions

  stringstream conv_1(argv[1]);
  stringstream conv_2(argv[2]);
  int n, thresh;
  if (!(conv_1 >> n))
    n = 4;
  if (!(conv_2 >> thresh))
    thresh = 1;
  // str << argv[1] << argv[2];
  // str >> n >> thresh;
  // str << argv[2];
  // str >> thresh;
  int N = pow(2,n);
  printf("n=%d // N=%d // thresh=%d:\n",n,N,thresh);

  // x = inputs, y = outputs 
  float *x;
  CUDA_SAFE_CALL(cudaMallocManaged(&x, N*sizeof(float)));
  printf("Init input:\n");
  for (int i=0; i<N; i++) {
    x[i]=(float) (N-i); // backwards list 
    // x[i]=(float) i; // sorted list
    // x[i]=(float) (rand() % 50); // random list 
    if (i<thresh || i>N-thresh-1)
      printf("for i=%d: x=%f\n", i, x[i]);
  }

  // TO-DO: 
  // -> get banyan_batcher function call working network
  // -> make sure args match those used in 'validation_nov' code 
  // -> ensure: does num_items == 2^n?
  banyan_batcher(x, N, thresh); // x = d_keys, N = num_items

  CUDA_SAFE_CALL(cudaFree(x));  

  // bitonic mergesort on batcher-banyan network
  // int level = 0;
  // int stage = 0;
  // int substage = 0;
  // int div;
  // int threadNum;
  // int compThreadNum = 512;
  // int compBlockNum = min(65535,(N+compThreadNum-1)/compThreadNum); // max(blockNum)=65535
  // printf("compareAndSwap blockNum=%d - threadNum=%d\n", compBlockNum, compThreadNum);
  // while (stage < n) {
  //   while (substage <= stage) { 
  //     div = N/(pow(2,2+stage-substage));
  //     // printf("stage=%d - substage=%d - div=%d - level=%d\n", stage, substage, div, level);
  //     compareAndSwap<<<compThreadNum,compBlockNum>>>(x, level, N);
  //     cudaDeviceSynchronize();
  //     // printf("-> compare for stage=%d at level=%d\n", stage, level);
  //     if (stage < n-1) {
  //       threadNum = min(1024, N/div); 
  //       if (substage == 0) {
  //         shuffleN<<<div,threadNum>>>(x, y, N/div, N);
  //         cudaDeviceSynchronize();
  //         // printf("-> shuffle for stage=%d\n", stage);
  //         level++;
  //       }
  //       compareAndSwap<<<compThreadNum,compBlockNum>>>(x, level, N);
  //       cudaDeviceSynchronize();
  //       // printf("-> compare for stage=%d at level=%d\n", stage, level);
  //       butterflyN<<<div,threadNum>>>(x, y, N/div, N);
  //       cudaDeviceSynchronize();
  //       // printf("-> butterfly for stage=%d, substage=%d\n", stage, substage);
  //       substage++;
  //     }
  //     else {
  //       break;
  //     }
  //   }
  //   stage++;
  //   substage = 0;
  // }
  // printf("-> sorted result for N=%d\n", N);
  // for (int i=0; i<N; i++)
  //   printf("for i=%d: x=%f\n", i, x[i]);

  // snippets for testing kernels 

  // -> compare and swap - VALIDATED 
  // int level = 1;
  // int threadNum = 512;
  // int blockNum = min(65535,(N+threadNum-1)/threadNum); // max(blockNum)=65535
  // compareAndSwap<<<blockNum,threadNum>>>(x, level, N);
  // cudaDeviceSynchronize();
  // printf("Result of COMPARE:\n");
  // // for (int i=0; i<N/2; i++)
  // //   printf("for i=%d: comparator=%d\n", i, comparators[i]);
  // for (int i=0; i<N; i++)
  //   printf("for i=%d: x=%f\n", i, x[i]);

  // -> getN - VALIDATED 
  // for (int size = 3; size < pow(2,8); size=size*size)  {
  //   printf("getN for size=%d: %d\n", size, getN(size));
  // }

  // -> butterflyN - VALIDATED 
  // int div = 1024;
  // int threadNum = min(1024, N/div);
  // butterflyN<<<div,threadNum>>>(x, y, N/div, N);
  // cudaDeviceSynchronize();
  // printf("Result of BUTTERFLY routing:\n");
  // for (int i=0; i<N; i++)
  //   printf("for i=%d: x=%f\n", i, x[i]);

  // -> shuffleN - VALIDATED 
  // int div = 1024;
  // int threadNum = min(1024, N/div);
  // shuffleN<<<div,threadNum>>>(x, y, N/div, N);
  // cudaDeviceSynchronize();
  // printf("Result of SHUFFLE routing:\n");
  // for (int i=0; i<N; i++)
  //   printf("for i=%d: x=%f\n", i, x[i]);

  return 0;                           
}
