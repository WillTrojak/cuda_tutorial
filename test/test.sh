#! /bin/bash

rm -f ._test_cuda
rm -f ._test_cuda.cu

cat  << EOF > ._test_cuda.cu

#include <stdio.h>
__global__ void test_kern(int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n)
       float x = 2.*i;
}

int main(){
    int n = 10;
    test_kern<<<128, 1>>>(n);
    printf("Good to go!\n");
    return 0;
}
EOF

nvcc -diag-suppress 177 ._test_cuda.cu -o ._test_cuda && ./._test_cuda
rm -f ._test_cuda
rm -f ._test_cuda.cu
