#include <stdio.h>

__global__ void hello() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main() {
    hello <<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}