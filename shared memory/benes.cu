#include <iostream>
#define UP 0 
#define DOWN 1

__device__ void compare_and_swith(int direction, float *in0, float *in1){
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

__device__ void shuffle(int numElements, float *array){

}
    
__device__ void butterfly(int numElements, float *array){

}

__global__ void stage_zero(){
    
}