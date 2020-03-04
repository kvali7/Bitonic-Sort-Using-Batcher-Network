#include <iostream>
#define UP 0 
#define DOWN 1

__device__ compare_and_swith(int direction, int *in0, int *in1){
    if (direction == UP){
        if(*in1 > *in0){
            int temp = *in0;
            *in0 = *in1;
            *in1 = temp;
        }
    }
    if (direction == DOWN){
        if(*in0 > *in1){
            int temp = *in0;
            *in0 = *in1;
            *in1 = temp;
        }
    }
}
__global__ void benesNet(float *inArray){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int in0 = inArray[index * 2];
    int in1 = inArray[index * 2 + 1];
    int stage = 0;
    while(true){
        // TODO: update the logic for assigning direction, 
        // outIndex0, and outIndex1
        int direction = stage % 2;
        int outIndex0 = (index << stage) % 2; 
        int outIndex1 = stage % 2;
        compare_and_swith(direction, &in0, &in1);
        __syncthreads();
        inArray[outIndex0] = in0;
        inArray[outIndex1] = in1; 

        ++stage;
        if (stage == 2 * n - 1)
            break;
    }
}
