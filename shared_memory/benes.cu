#include <iostream>
#define UP 0 
#define DOWN 1

__device__ void compare_and_switch(int direction, float *in0, float *in1){
    if (direction == UP){
        if(*in1 > *in0){
            float temp = *in0;
            *in0 = *in1;
            *in1 = temp;
        }
    }
    if (direction == DOWN){
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
    printf("stageSize: %8d\tvalue: %f\naddrOut: %8d\toffset: %4d\narray[%d + %d] = %f\n\n", stageSize, value, addrOut, offset, addrIn, offset, value);
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
    array[addrIn + offset] = value;
}

__global__ void stagingKernel(int stageNum, int stageSize, int numElements, float *arrayIn){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 
    int rectNum = threadIdx.x / (stageSize / 2);
    extern __shared__ float buffer[];

    int level = stageNum; 

    for (int addr = index; addr < (numElements / 2); addr += stride){   
        
        
        int outAddrOffset = addr - (addr % (stageSize / 2));
        int outAddr0 = (addr * 2) % stageSize;
        int outAddr1 = (addr * 2 + 1) % stageSize;
    
        float firstIn = arrayIn[addr * 2];
        float secondIn = arrayIn[addr * 2 + 1];   

        // printf("addr: %8d\toutAddrOffset: %4d\noutAddr0: %4d\toutAddr1: %9d\nfirstIn: %f\tsecondIn: %f\n\n", addr, outAddrOffset, outAddr0, outAddr1, firstIn, secondIn);
        // printf ("addr: %4d\trectNum: %4d\n", addr, rectNum);

        int direction = (addr & (1 << level)) == 0 ? UP : DOWN;
        compare_and_switch(direction, &firstIn, &secondIn);
        
        shuffle(stageSize, firstIn, outAddr0, outAddrOffset, buffer);
        shuffle(stageSize, secondIn, outAddr1, outAddrOffset, buffer);
    }

    level++;

    for (int iteration = 0; iteration <= stageNum; ++iteration){    
        for (int addr = index; addr < (numElements / 2); addr += stride){
            int direction = addr & (1 << level) == 0 ? UP : DOWN;
            int outAddrOffset = addr - (addr % (stageSize / 2));
            int outAddr0 = (addr * 2) % stageSize;
            int outAddr1 = (addr * 2 + 1) % stageSize;
            float firstIn = buffer[tIndex * 2];
            float secondIn = buffer[tIndex * 2 + 1];
            
            compare_and_switch(direction, &firstIn, &secondIn);
            butterfly(stageSize, firstIn, outAddr0, outAddrOffset, buffer);
            butterfly(stageSize, secondIn, outAddr1, outAddrOffset, buffer); 
        }
        // arrayIn[addr * 2] = buffer[tIndex * 2];
        // arrayIn[addr * 2 + 1] = buffer[tIndex * 2 + 1];
    }

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
    for (int stageNum = 0; stageNum < n; stageNum++){
        printf("stageNum: %4d\t stageSize: %4d\n\n", stageNum, stageSize);
        stagingKernel<<< 64, 1024, stageSize * sizeof(float)>>>(stageNum, stageSize, N, x);
        cudaDeviceSynchronize();
        printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        stageSize = stageSize * 2;
    }
    cleanupKenerl<<< 80, 1024>>>(N, x);
    
}