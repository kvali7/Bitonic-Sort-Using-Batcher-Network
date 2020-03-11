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
void shuffleN(float *in, float *out, int size, ulong N) {
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
void butterflyN(float *in, float *out, int size, ulong N) {
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
void compareAndSwap(float *in, int level, ulong N) {
  // TO-DO: pull in shared memory
  unsigned long stride = blockDim.x*gridDim.x; 
  // if (comp_ind < N) {
  for (unsigned long comp_ind = (unsigned long) blockDim.x*blockIdx.x+threadIdx.x;
       comp_ind < N/2;
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

void banyan(float *x, float *y, ulong N, uint n) {
  // bitonic mergesort on batcher-banyan network
  printf("banyan_batcher in function call: N=%d - n=%d\n",(int)N,(int)n);
  int level = 0;
  int stage = 0;
  int substage = 0;
  int div; // blockNum for current routing kernel
  int threadNum; // threadNum for current routing kernel
  int compThreadNum = 512; // threadNum for compareAndSwap kernel
  int compBlockNum = min((long)65535,(N+compThreadNum-1)/compThreadNum); // max(blockNum)=65535

  CUDA_SAFE_CALL(cudaMallocManaged(&y, N*sizeof(float)));
  // printf("compareAndSwap blockNum=%d - threadNum=%d\n", compBlockNum, compThreadNum);
  while (stage < n) {
    while (substage <= stage) { 
      div = N/(pow(2,2+stage-substage));
      // printf("stage=%d - substage=%d - div=%d - level=%d\n", stage, substage, div, level);
      if (stage < n-1) {
        threadNum = min((long)1024, N/div); 
        if (substage == 0) {
          compareAndSwap<<<compBlockNum,compThreadNum>>>(x, level, N);
          cudaDeviceSynchronize();
          // printf("-> compare for stage=%d at level=%d\n", stage, level);
          shuffleN<<<div,threadNum>>>(x, y, N/div, N);
          cudaDeviceSynchronize();
          // printf("-> shuffle for stage=%d\n", stage);
          level++;
        }
        compareAndSwap<<<compBlockNum,compThreadNum>>>(x, level, N);
        cudaDeviceSynchronize();
        // printf("-> compare for stage=%d at level=%d\n", stage, level);
        butterflyN<<<div,threadNum>>>(x, y, N/div, N);
        cudaDeviceSynchronize();
        // printf("-> butterfly for stage=%d, substage=%d\n", stage, substage);
        substage++;
      }
      else {
        compareAndSwap<<<compBlockNum,compThreadNum>>>(x, level, N);
        cudaDeviceSynchronize();
        // printf("-> final compare for stage=%d at level=%d\n", stage, level);
        break;
      }
    }
    stage++;
    substage = 0;
  }
  CUDA_SAFE_CALL(cudaFree(y));
  // printf("-> first and last elements of sorted result for N=%d\n", (int)N);
  // for (int i=0; i<2*thresh-1; i++)
  //   printf("for i=%d: x=%f\n", (i<thresh) ? i : (int)N-(2*thresh-i-1) , (i<thresh) ? x[i] : x[(int)N-(2*thresh-i-1)]);
}

// main for debugging individual kernels
// int main(int argc, char** argv)
// {
//   // USAGE: single argument
//   // -> n = argv[1]
//   // --> e.g. "./banyan 4" would run n=4, N=16
// 
//   // params for testing helper functions
// 
//   stringstream conv_1(argv[1]);
//   stringstream conv_2(argv[2]);
//   uint n;
//   int thresh;
//   if (!(conv_1 >> n))
//     n = 4;
//   if (!(conv_2 >> thresh))
//     thresh = 1;
//   ulong N = pow(2,n);
//   printf("n=%d // N=%d // thresh=%d:\n",(int)n,(int)N,thresh); // NOTE: might not be exposing issue by casting to int here
// 
//   // x = inputs, y = outputs 
//   float *x, *y;
//   CUDA_SAFE_CALL(cudaMallocManaged(&x, N*sizeof(float)));
//   CUDA_SAFE_CALL(cudaMallocManaged(&y, N*sizeof(float)));
//   printf("Init input:\n");
//   for (int i=0; i<N; i++) {
//     x[i]=(float) (N-i); // backwards list 
//     // x[i]=(float) i; // sorted list
//     // x[i]=(float) (rand() % 50); // random list 
//     if (i<thresh || i>N-thresh-1)
//       printf("for i=%d: x=%f\n", i, x[i]);
//   }
// 
//   // call batcher-banyan sorting network on N-element array
//   banyan(x, y, N, n); // x = d_keys, N = num_items
// 
//   // snippets for testing kernels 
// 
//   // -> compare and swap - VALIDATED UP TO n=21
//   // int level = 0;
//   // int threadNum = 512;
//   // int blockNum = min((long)65535,(N+threadNum-1)/threadNum); // max(blockNum)=65535
//   // printf("Test COMPARE AND SWAP for N=%d with %d threads per %d blocks:\n",(int)N, threadNum, blockNum);
//   // compareAndSwap<<<blockNum,threadNum>>>(x, level, N);
//   // cudaDeviceSynchronize();
//   // printf("Result of COMPARE AND SWAP for N=%d:\n",(int)N);
//   // // for (int i=0; i<N; i++)
//   // //   printf("for i=%d: x=%f\n", i, x[i]);
//   // for (int i=0; i<2*thresh-1; i++)
//   //   printf("for i=%d: x=%f\n", (i<thresh) ? i : (int)N-(2*thresh-i-1) , (i<thresh) ? x[i] : x[(int)N-(2*thresh-i-1)]);
// 
//   // -> getN - NOT VALIDATED 
//   // for (int size = 3; size < pow(2,8); size=size*size)  {
//   //   printf("getN for size=%d: %d\n", size, getN(size));
//   // }
// 
//   // -> butterflyN - VALIDATED UP TO n=20 
//   // float *y;
//   // CUDA_SAFE_CALL(cudaMallocManaged(&y, N*sizeof(float)));
//   // int div = N/8;
//   // int threadNum = min((long)1024, N/div);
//   // printf("Test BUTTERFLY for N=%d with %d threads per %d blocks:\n",(int)N, threadNum, div);
//   // butterflyN<<<div,threadNum>>>(x, y, N/div, N);
//   // cudaDeviceSynchronize();
//   // printf("Result of BUTTERFLY for N=%d:\n",(int)N);
//   // for (int i=0; i<2*thresh-1; i++)
//   //   printf("for i=%d: x=%f\n", (i<thresh) ? i : (int)N-(2*thresh-i-1) , (i<thresh) ? x[i] : x[(int)N-(2*thresh-i-1)]);
// 
//   // -> shuffleN - VALIDATED UP TO n=20 
//   // float *y;
//   // CUDA_SAFE_CALL(cudaMallocManaged(&y, N*sizeof(float)));
//   // int div = N/4;
//   // int threadNum = min((long)1024, N/div);
//   // printf("Test SHUFFLE for N=%d with %d threads per %d blocks:\n",(int)N, threadNum, div);
//   // shuffleN<<<div,threadNum>>>(x, y, N/div, N);
//   // cudaDeviceSynchronize();
//   // printf("Result of SHUFFLE for N=%d:\n",(int)N);
//   // for (int i=0; i<2*thresh-1; i++)
//   //   printf("for i=%d: x=%f\n", (i<thresh) ? i : (int)N-(2*thresh-i-1) , (i<thresh) ? x[i] : x[(int)N-(2*thresh-i-1)]);
// 
//   CUDA_SAFE_CALL(cudaFree(x));  
//   CUDA_SAFE_CALL(cudaFree(y));  
// 
//   return 0;                           
// }
