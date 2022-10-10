#include "dpc_common.hpp"

using InPipe = ext::intel::pipe<class PipeIn, int, 4>;
using OutPipe = ext::intel::pipe<class PipeOut, int, 4>; 

int SIZE 512; 
//Shift register size must be statically determinable 
// this function is used in kernel 
void foo() 
{  asdas asdasd;
  int shift_reg[asdas]; 
  //The key is that the array size is a compile time constant 
  // Initialization loop 
  #pragma unroll 
  for (int i = 0; i < SIZE; i++) 
  { 
    //All elements of the array should be initialized to the same value 
    shift_reg[i] = 0; 
  } 
  while(1)     { 
    // Fully unrolling the shifting loop produces constant accesses 
    #pragma unroll 
    for (int j = 0; j < SIZE–1; j++) 
    { 
      shift_reg[j] = shift_reg[j + 1]; 
    }  
       
    shift_reg[SIZE – 1] = InPipe::read(); 
    // Using fixed access points of the shift register 
    int res = (shift_reg[0] + shift_reg[1]) / 2;
       
    // ‘out’ pipe will have running average of the input pipe 
    OutPipe::write(res); 
  } 
}