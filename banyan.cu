#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
using namespace std;
// Kernel function to add the elements of two arrays
__global__
void copy_vec(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; // contains index of current thread
  int stride = blockDim.x * gridDim.x; // contains number of threads in block
  for (int i = index; i < n/4; i+=stride)
    reinterpret_cast<float4*>(y)[i] = reinterpret_cast<float4*>(x)[i];
}

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
void shuffleN(float *in, float *out, int size) {
  // TO-DO: pull in shared memory
  int ind_in  = threadIdx.x;
  int ind_out; 
  // ind_out = 2*(ind_in-ind_in % size/2);
  if (ind_in % 2 == 0) { // even 
    ind_out = ind_in/2;
  }  
  else {  // odd 
    ind_out = (size/2)+(ind_in/2);
  }
  out[ind_in]=in[ind_out];
}

// BUTTERFLY kernel for N=pow(2,n) elements
__global__
void butterflyN(float *in, float *out, int size) {
  // TO-DO: pull in shared memory
  int ind_in  = threadIdx.x;
  int ind_out; 
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
  out[ind_in]=in[ind_out];
}

// TO-DO: Implement COMPARE kernel (NOT VALIDATED!)
__global__
void compare(float *in, float *out, bool *directions, int level, int size) {
  // TO-DO: pull in shared memory
  int ind_in  = blockDim.x*blockIdx.x+threadIdx.x;
  // STEP 0: Get direction of comparison
  printf("ind_in: %d\n", ind_in);
  bool comp_bool = (bool)((char) ind_in && (0x1 << level)); 
  printf("for ind_in=%d comp_bool=%d\n", ind_in, comp_bool);
  // if (ind_in < size/2) { // first half of list
  //   if (ind_in%2==0) // even
  //     ind_out = ind_in;
  //   else // odd
  //     ind_out = ind_in + (size/2 - 1);
  // }
  directions[ind_in] = comp_bool;
  // STEP 1: Write out results of comparisons
  // out[ind_in]=in[ind_out];
}

int main(int argc, char** argv)
{
  // params for testing helper functions
  int level = 0;
  int n = 4;
  int N = pow(2,n);
  printf("n=%d // N=%d:\n",n,N);

  // x = inputs, y = outputs 
  float *x, *y;
  bool *comparators; 
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&comparators, N/2*sizeof(bool));
  printf("Init input:\n");
  for (int i=0; i<N; i++) {
    x[i]=(float) i;
  }

  // test helper functions

  // -> compare - NOT VALIDATED 
  compare<<<1,N/2>>>(x, y, comparators, level, N);
  cudaDeviceSynchronize();
  printf("Result of COMPARE:\n");
  for (int i=0; i<N/2; i++)
    printf("for i=%d: comparator=%d\n", i, comparators[i]);

  // -> getN - VALIDATED 
  // for (int size = 3; size < pow(2,8); size=size*size)  {
  //   printf("getN for size=%d: %d\n", size, getN(size));
  // }

  // -> butterflyN - VALIDATED 
  // butterflyN<<<1,N>>>(x, y, N);
  // cudaDeviceSynchronize();
  // printf("Result of BUTTERFLY routing:\n");
  // for (int i=0; i<N; i++)
  //   printf("for i=%d: x=%f -> y=%f\n", i, x[i], y[i]);

  // -> shuffleN - VALIDATED 
  // shuffleN<<<1,N>>>(x, y, N);
  // cudaDeviceSynchronize();
  // printf("Result of SHUFFLE routing:\n");
  // for (int i=0; i<N; i++)
  //   printf("for i=%d: x=%f -> y=%f\n", i, x[i], y[i]);

  cudaFree(x);  
  cudaFree(y);
  cudaFree(comparators);

  return 0;
}
