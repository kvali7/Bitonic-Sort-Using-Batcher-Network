#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <thrust/swap.h>
using namespace std;

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

// DEVICE helper function: reverse a string  */
__device__
void reverse(char str[], int length) 
{ 
    int start = 0; 
    int end = length -1; 
    while (start < end) 
    { 
        // swap(*(str+start), *(str+end)); 
        thrust::swap(*(str+start), *(str+end)); 
        start++; 
        end--; 
    } 
} 

// DEVICE helper function: Implementation of itoa() 
__device__
char* itoa(int num, char* str, int base) 
{ 
    int i = 0; 
    bool isNegative = false; 
  
    /* Handle 0 explicitely, otherwise empty string is
 * printed for 0 */
    if (num == 0) 
    { 
        str[i++] = '0'; 
        str[i] = '\0'; 
        return str; 
    } 
  
    // In standard itoa(), negative numbers are handled
    // only with  
    // base 10. Otherwise numbers are considered
    // unsigned. 
    if (num < 0 && base == 10) 
    { 
        isNegative = true; 
        num = -num; 
    } 
  
    // Process individual digits 
    while (num != 0) 
    { 
        int rem = num % base; 
        str[i++] = (rem > 9)? (rem-10) + 'a' : rem + '0'; 
        num = num/base; 
    } 
  
    // If number is negative, append '-' 
    if (isNegative) 
        str[i++] = '-'; 
  
    str[i] = '\0'; // Append string terminator 
  
    // Reverse the string 
    reverse(str, i); 
  
    return str; 
} 

// DEVICE helper function: print char (from void
__device__
void printbinchar(char character)
{
    char output[9];
    itoa(character, output, 2);
    printf("%s\n", output);
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

// TO-DO: Implement COMPARE kernel (NOT VALIDATED FOR LARGER BLOCKNUM, THREADNUM!)
__global__
void compare(float *in, bool *directions, int level, int N) {
  // TO-DO: pull in shared memory
  unsigned long stride = blockDim.x*gridDim.x; 
  // if (comp_ind < N) {
  for (unsigned long comp_ind = (unsigned long) blockDim.x*blockIdx.x+threadIdx.x;
       comp_ind < N;
       comp_ind += stride) {
    // STEP 0: Get direction of comparison
    // printf("ind_in: %d\n", ind_in);
    // printbinchar((char) ind_in & (0x1 << level));
    bool comp_bool = (bool)(comp_ind & (0x1 << level)); 
    // printf("for ind_in=%d comp_bool=%d\n", ind_in, comp_bool);
    directions[comp_ind] = comp_bool;
    // STEP 1: Write out results of comparisons
    unsigned long data0 = 2*comp_ind;
    unsigned long data1 = 2*comp_ind+1;
    // compare and swap based on comp_bool
    if ( (comp_bool && in[data0] < in[data1]) || (!comp_bool && in[data0] >= in[data1]) ) { 
        thrust::swap(*(in+data0), *(in+data1)); 
    }
  }
}

int main(int argc, char** argv)
{
  // params for testing helper functions
  stringstream str;
  str << argv[1];
  int n;
  str >> n;
  // int n = 10;
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
    x[i]=(float) (N-i);
    // x[i]=(float) i;
    printf("for i=%d: x=%f\n", i, x[i]);
  }

  // bitonic mergesort on batcher-banyan network
  int level = 0;
  int stage = 0;
  int substage = 0;
  int div;
  int threadNum;
  int compThreadNum = 512;
  int compBlockNum = min(65535,(N+compThreadNum-1)/compThreadNum); // max(blockNum)=65535
  printf("comparator blockNum=%d - threadNum=%d\n", compBlockNum, compThreadNum);
  while (stage < n) {
    while (substage <= stage) { 
      div = N/(pow(2,2+stage-substage));
      printf("stage=%d - substage=%d - div=%d - level=%d\n", stage, substage, div, level);
      compare<<<compThreadNum,compBlockNum>>>(x, comparators, level, N);
      cudaDeviceSynchronize();
      printf("-> compare for stage=%d at level=%d\n", stage, level);
      // for (int i=0; i<N; i++)
      //   printf("for i=%d: x=%f\n", i, x[i]);
      if (stage < n-1) {
        threadNum = min(1024, N/div); 
        if (substage == 0) {
          shuffleN<<<div,threadNum>>>(x, y, N/div, N);
          cudaDeviceSynchronize();
          // printf("-> shuffle for stage=%d\n", stage);
          // for (int i=0; i<N; i++)
          //   printf("for i=%d: x=%f\n", i, x[i]);
          level++;
        }
        compare<<<compThreadNum,compBlockNum>>>(x, comparators, level, N);
        cudaDeviceSynchronize();
        // printf("-> compare for stage=%d at level=%d\n", stage, level);
        // for (int i=0; i<N; i++)
        //   printf("for i=%d: x=%f\n", i, x[i]);
        butterflyN<<<div,threadNum>>>(x, y, N/div, N);
        cudaDeviceSynchronize();
        // printf("-> butterfly for stage=%d, substage=%d\n", stage, substage);
        // for (int i=0; i<N; i++)
        //   printf("for i=%d: x=%f\n", i, x[i]);
        substage++;
      }
      else {
        break;
      }
    }
    stage++;
    substage = 0;
  }
  printf("-> sorted result for N=%d\n", N);
  for (int i=0; i<N; i++)
    printf("for i=%d: x=%f\n", i, x[i]);

  // test helper functions

  // -> compare and swap - VALIDATED 
  // int level = 1;
  // int threadNum = 512;
  // int blockNum = min(65535,(N+threadNum-1)/threadNum); // max(blockNum)=65535
  // compare<<<blockNum,threadNum>>>(x, comparators, level, N);
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

  cudaFree(x);  
  cudaFree(y);
  cudaFree(comparators);

  return 0;
}
