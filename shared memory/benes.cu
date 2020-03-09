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

__device__ void shuffle(int stageSize, int value, int addrOut, int offset, float *array){
    // Actually destination address is addrIn, 
    // it may look unintuitive but the fact is
    // In N' Out (pun intended with the california
    // based fast food chain) are relative terms based 
    // on where you're looking at the code from. In this case 
    // the shuffle function in going to caluculate the input address 
    // (addrIn) for the shared memory based on the output address of
    // the previous comparator (addrOut)
    int addrIn = addrOut; 
    addrIn << 1;
    addrIn &= ~(stageSize);
    addrIn |= (addrIn / (stageSize / 2));
    array[addrIn + offset] = value;
}

__device__ void butterfly(int stageSize, int value, int addrOut, int offset, float *array){

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
    int tIndex = threadIdx.x; 

    extern __shared__ float buffer[];

    for (int addr = index; addr < (numElements / 2); addr += stride){
        int level = stageNum; 
        int direction = -1;

        int outAddrOffset = tIndex - (tIndex % stageSize);
        int outAddr0 = (addr * 2) % stageSize;
        int outAddr1 = (addr * 2 + 1) % stageSize;
    
        float firstIn = arrayIn[addr * 2];
        float secondIn = arrayIn[addr * 2 + 1];
        
        direction = (addr & (1 << level)) == 0 ? UP : DOWN;
        compare_and_switch(direction, &firstIn, &secondIn);
        
        shuffle(stageSize, firstIn, outAddr0, outAddrOffset, buffer);
        shuffle(stageSize, secondIn, outAddr1, outAddrOffset, buffer);
        
        level++;
        direction = addr & (1 << level) == 0 ? UP : DOWN;
        
        for (int iteration = 0; iteration <= stageNum; ++iteration){
            firstIn = buffer[tIndex * 2];
            secondIn = buffer[tIndex * 2 + 1];
            compare_and_switch(direction, &firstIn, &secondIn);
            butterfly(stageSize, firstIn, outAddr0, outAddrOffset, buffer);
            butterfly(stageSize, secondIn, outAddr1, outAddrOffset, buffer); 
        }
    }

}

int main (void)