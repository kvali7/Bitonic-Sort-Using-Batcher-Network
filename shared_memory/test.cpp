#include <iostream> 

using namespace std; 

int main (void){
    unsigned int stageSize = 128;
    for (int index = 0; index < stageSize; index++){
        int addrOut = index;
        printf("addrOut: %2d", addrOut); 
        int firstBit = addrOut % 2; 
        int lastBit = addrOut / (stageSize / 2);
        int addrIn = addrOut; 
        addrIn = addrIn - (addrIn % 2);
        addrIn &= ~(stageSize / 2);
        addrIn |= (firstBit * stageSize / 2);
        addrIn |= lastBit;
        printf("\taddrIn: %2d\n", addrIn); 
    }
    return 0;
}