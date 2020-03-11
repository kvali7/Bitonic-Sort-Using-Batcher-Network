#include <iostream>
#include <sstream>
#include <stdio.h>
using namespace std;

#define UP 0
#define DOWN 1

#define BLOCK_SIZE 1024
#define NUM_BLOCKS 128
#define SHARED_MEM 8192

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

__device__ int getButtSize(int stageSize, int numButterfly){
    return stageSize >> numButterfly;
}
__device__ int maximum(int firstIn, int secondIn){
    return firstIn >= secondIn ? firstIn : secondIn;
}

__device__ void compare_and_switch(int direction, float *in0, float *in1){
    if (direction == DOWN){
        // printf("direction: UP\tin0: %f\tin1: %f\n", *in0, *in1);
        if(*in1 > *in0){
            float temp = *in0;
            *in0 = *in1;
            *in1 = temp;
        }
    }
    if (direction == UP){
        // printf("direction: DN\tin0: %f\tin1: %f\n", *in0, *in1);
        if(*in0 > *in1){
            float temp = *in0;
            *in0 = *in1;
            *in1 = temp;
        }
    }
}

__device__ void shuffle(int stageSize, float value, int addrOut, int offset, float *array){
    // Actually destination address is addrIn, 
    // it may look unintuitive but the fact is
    // In N' Out (pun intended with the california
    // based fast food chain) are relative terms based 
    // on where you're looking at the code from. In this case 
    // the shuffle function in going to caluculate the input address 
    // (addrIn) for the shared memory based on the output address of
    // the previous comparator (addrOut)
    int addrIn = addrOut; 
    addrIn = addrIn << 1;
    addrIn &= ~(stageSize);
    addrIn |= (addrOut / (stageSize / 2));
    // printf("stageSize: %8d\tvalue: %f\naddrOut: %8d\toffset: %4d\narray[%d + %d] = %f\n\n", stageSize, value, addrOut, offset, addrIn, offset, value);
    array[addrIn + offset] = value;
}

__device__ void butterfly(int stageSize, float value, int addrOut, int offset, float *array){

    int firstBit = addrOut % 2; 
    int lastBit = addrOut / (stageSize / 2);
    int addrIn = addrOut; 
    addrIn = addrIn - (addrIn % 2);
    addrIn &= ~(stageSize / 2);
    addrIn |= (firstBit * stageSize / 2);
    addrIn |= lastBit;
    // printf("stageSize: %8d\tvalue: %f\naddrOut: %8d\toffset: %4d\narray[%d + %d] = %f\n\n", stageSize, value, addrOut, offset, addrIn, offset, value);
    array[addrIn + offset] = value;
}

__global__ void stagingKernel(int stageNum, int stageSize, int numElements, float *arrayIn){

    int quotient = maximum(blockDim.x, stageSize / 2);
    int index = blockIdx.x * quotient + threadIdx.x; 
    int stride = gridDim.x * quotient; 
    __shared__ float buffer[SHARED_MEM];

    int level = stageNum; 
    // printf ("blockIdx.x: %4d\tthreadIdx.x: %4d\nindex: %4d\tstride: %4d\n\n", blockIdx.x, threadIdx.x, index, stride);
    for (int addr = index; addr < (numElements / 2); addr += stride){   
        for (int iteration = 0; iteration < maximum(stageSize / (2 * BLOCK_SIZE), 1); ++iteration){
            int compGlobalAddr = addr + iteration * BLOCK_SIZE;
            int compLocalAddr = threadIdx.x + iteration * BLOCK_SIZE;
            // printf ("blockIdx.x: %4d\tthreadIdx.x: %4d\ncompGlobalAddr: %4d\tcompLocalAddr: %4d\n\n", blockIdx.x, threadIdx.x, compGlobalAddr, compLocalAddr);

            int firstInAddr = compGlobalAddr * 2;
            int secondInAddr = compGlobalAddr * 2 + 1;
            float firstIn = arrayIn[firstInAddr];
            float secondIn = arrayIn[secondInAddr];
            
            int outAddrOffset = (compLocalAddr * 2 / stageSize) * stageSize;
            int firstOutAddr = (compLocalAddr * 2) % stageSize;
            int secondOutAddr = (compLocalAddr * 2 + 1) % stageSize;  

            // printf("compGlobalAddr: %8d\tcompLocalAddr: %4d\noutAddrOffset: %4d\tfirstoutAddr: %4d\tsecondOutAddr: %9d\nfirstIn: %f\tsecondIn: %f\n\n", compGlobalAddr, compLocalAddr, outAddrOffset, firstOutAddr, secondOutAddr, firstIn, secondIn);
         

            int direction = (compGlobalAddr & (1 << level)) == 0 ? UP : DOWN;
            // printf("compGlobalAddr: %4d\tdirection: %d\n", compGlobalAddr, direction);
            compare_and_switch(direction, &firstIn, &secondIn);
        
            shuffle(stageSize, firstIn, firstOutAddr, outAddrOffset, buffer);
            shuffle(stageSize, secondIn, secondOutAddr, outAddrOffset, buffer);
        }

        level++;

        // if (threadIdx.x == 0)
        //     printf("blockIdx.x: %d\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n\n", blockIdx.x,
        //                                                                                buffer[0], buffer[1], buffer[2], buffer[3], 
        //                                                                                buffer[4], buffer[5], buffer[6], buffer[7], 
        //                                                                                buffer[8], buffer[9], buffer[10], buffer[11], 
        //                                                                                buffer[12], buffer[13], buffer[14], buffer[15]);
        // __syncthreads();
        
        for (int iteration = 0; iteration < maximum(stageSize / (2 * BLOCK_SIZE), 1); ++iteration){
            for (int numButterfly = 0; numButterfly <= stageNum; ++numButterfly){

                int butterflySize = getButtSize(stageSize, numButterfly);
                int compGlobalAddr = addr + iteration * BLOCK_SIZE;
                int compLocalAddr = threadIdx.x + iteration * BLOCK_SIZE;

                int firstInAddr = compLocalAddr * 2;
                int secondInAddr = compLocalAddr * 2 + 1;
                float firstIn = buffer[firstInAddr];
                float secondIn = buffer[secondInAddr];   
                
                int outAddrOffset = (compLocalAddr * 2 / butterflySize) * butterflySize;
                int firstOutAddr = (compLocalAddr * 2) % butterflySize;
                int secondOutAddr = (compLocalAddr * 2 + 1) % butterflySize;

                int direction = (compGlobalAddr & (1 << level)) == 0 ? UP : DOWN;
                compare_and_switch(direction, &firstIn, &secondIn);

                butterfly(butterflySize, firstIn, firstOutAddr, outAddrOffset, buffer);
                butterfly(butterflySize, secondIn, secondOutAddr, outAddrOffset, buffer);
               
            }
            __syncthreads();
        }

        // if (threadIdx.x == 0)
        //     printf("blockIdx.x: %d\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n\n", blockIdx.x,
        //                                                                                buffer[0], buffer[1], buffer[2], buffer[3], 
        //                                                                                buffer[4], buffer[5], buffer[6], buffer[7], 
        //                                                                                buffer[8], buffer[9], buffer[10], buffer[11], 
        //                                                                                buffer[12], buffer[13], buffer[14], buffer[15]);
        // __syncthreads();

        for (int iteration = 0; iteration < maximum(stageSize / (2 * BLOCK_SIZE), 1); ++iteration){
            int compGlobalAddr = addr + iteration * BLOCK_SIZE;
            int compLocalAddr = threadIdx.x + iteration * BLOCK_SIZE;

            int firstLocalAddr = compLocalAddr * 2;
            int secondLocalAddr = compLocalAddr * 2 + 1;

            int firstGlobalAddr = compGlobalAddr * 2;
            int secondGlobalAddr = compGlobalAddr * 2 + 1;
            // printf("compGlobalAddr: %d\tcompLocalAddr: %d\nbuffer[%d]: %f\tbuffer[%d]: %f\narrayIn[%d] = buffer[%d] = %f\narrayIn[%d] = buffer[%d] = %f\n\n", compGlobalAddr, compLocalAddr, firstLocalAddr, buffer[firstLocalAddr], secondLocalAddr, buffer[secondLocalAddr], firstGlobalAddr, firstLocalAddr, buffer[firstLocalAddr], secondGlobalAddr, secondLocalAddr, buffer[secondLocalAddr]);
            // printf("compGlobalAddr: %d\tcompLocalAddr: %d\nbuffer[%d]: %f\tbuffer[%d]: %f\narrayIn[%d] = buffer[%d] = %f\narrayIn[%d] = buffer[%d] = %f\n\n", compGlobalAddr, compLocalAddr, firstLocalAddr, secondLocalAddr, firstGlobalAddr, firstLocalAddr, buffer[firstLocalAddr], secondGlobalAddr, secondLocalAddr, buffer[secondLocalAddr]);
            arrayIn[firstGlobalAddr] = buffer[firstLocalAddr];
            arrayIn[secondGlobalAddr] = buffer[secondLocalAddr];
        }

        // if (threadIdx.x == 0)
        //     printf("%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n\n", buffer[0], buffer[1], buffer[2], buffer[3], 
        //                                                                                buffer[4], buffer[5], buffer[6], buffer[7], 
        //                                                                                buffer[8], buffer[9], buffer[10], buffer[11], 
        //                                                                                buffer[12], buffer[13], buffer[14], buffer[15]);
        // __syncthreads();
    }

    // level++;

    // for (int iteration = 0; iteration <= stageNum; ++iteration){    
        // for (int addr = index; addr < (numElements / 2); addr += stride){
            
            // int direction = addr & (1 << level) == 0 ? UP : DOWN;

    //         int outAddrOffset = addr - (addr % (stageSize / 2));
    //         int outAddr0 = (addr * 2) % stageSize;
    //         int outAddr1 = (addr * 2 + 1) % stageSize;
            
    //         float firstIn = buffer[tIndex * 2];
    //         float secondIn = buffer[tIndex * 2 + 1];
            
    //         compare_and_switch(direction, &firstIn, &secondIn);
    //         butterfly(stageSize, firstIn, outAddr0, outAddrOffset, buffer);
    //         butterfly(stageSize, secondIn, outAddr1, outAddrOffset, buffer); 
    //     }
    //     // arrayIn[addr * 2] = buffer[tIndex * 2];
    //     // arrayIn[addr * 2 + 1] = buffer[tIndex * 2 + 1];
    // }

}

__global__ void cleanupKenerl(int numElements, float *arrayIn){
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 

    for (int addr = index; addr < (numElements / 2); addr += stride){
        float firstIn = arrayIn[addr * 2];
        float secondIn = arrayIn[addr * 2 + 1];
        compare_and_switch(UP, &firstIn, &secondIn);
        arrayIn[addr * 2] = firstIn;
        arrayIn[addr * 2 + 1] = secondIn;
    }
}

void banyan(float *x, ulong N, uint n){
    int stageSize = 4; 
    for (int stageNum = 0; stageNum < n - 1; stageNum++){
        // printf("stageNum: %4d\t stageSize: %4d\n\n", stageNum, stageSize);
        // stagingKernel<<< 1, 1, 16384 * sizeof(float)>>>(stageNum, stageSize, N, x);
        stagingKernel<<< NUM_BLOCKS, BLOCK_SIZE >>>(stageNum, stageSize, N, x);
        cudaDeviceSynchronize();
        // printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        stageSize = stageSize * 2;
    }
    cleanupKenerl<<< 80, 1024>>>(N, x);
    cudaDeviceSynchronize();
    
}

// main for debugging individual kernels
int main(int argc, char** argv)
{
  // USAGE: single argument
  // -> n = argv[1]
  // --> e.g. "./banyan 4" would run n=4, N=16

  // params for testing helper functions

  stringstream conv_1(argv[1]);
  stringstream conv_2(argv[2]);
  uint n;
  int thresh;
  if (!(conv_1 >> n))
    n = 4;
  if (!(conv_2 >> thresh))
    thresh = 1;
  ulong N = pow(2,n);
  printf("n=%d // N=%d // thresh=%d:\n",(int)n,(int)N,thresh); // NOTE: might not be exposing issue by casting to int here

  // x = inputs, y = outputs 
  float *x;
  CUDA_SAFE_CALL(cudaMallocManaged(&x, N*sizeof(float)));
  printf("------------------------------------------------------------\n");
  printf("Init input:\n");
  printf("------------------------------------------------------------\n");
  for (int i=0; i<N; i++) {
    x[i]=(float) (N-i-1); // backwards list 
    // x[i]=(float) i; // sorted list
    // x[i]=(float) (rand() % 50); // random list 
    if (i<thresh || i>N-thresh-1)
      printf("for i=%d: x=%f\n", i, x[i]);
  }

  // call batcher-banyan sorting network on N-element array
  banyan(x, N, n);
  cudaDeviceSynchronize();
  printf("------------------------------------------------------------\n");
  printf("Output:\n");
  printf("------------------------------------------------------------\n");
  for (int i=0; i<2*thresh-1; i++) {
    printf("for i=%d: x=%f\n", (i<thresh) ? i : (int)N-(2*thresh-i-1) , (i<thresh) ? x[i] : x[(int)N-(2*thresh-i-1)]);
  }

  CUDA_SAFE_CALL(cudaFree(x));  

  return 0;                           
}
