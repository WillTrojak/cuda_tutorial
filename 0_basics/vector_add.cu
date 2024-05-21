#include <stdio.h>

#include <cuda_runtime.h>

void error_check(cudaError_t cerr, int id) {
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Failed (%d) with error code %s\n", id, cudaGetErrorString(cerr));
        exit(EXIT_FAILURE);
    }
}


__global__ void vector_add(int n, const float *a, const float *b, float *c) {
    // -> Element indexing

    // -> C = A + B

}


int main(void) {
    cudaError_t cerr;

    int n = 50000;
    size_t size = n * sizeof(float);

    // Allocate the host memory
    float *a_h, *b_h, *c_h;

    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    c_h = (float *)malloc(size);

    // Initialise the vectors on the host
    for (int i = 0; i < n; ++i) {
        a_h[i] = rand() / (float)RAND_MAX;
        b_h[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device vectors
    float *a_d, *b_d, *c_d;
    cerr = cudaMalloc((void **)&a_d, size);
    error_check(cerr, 0);

    cerr = cudaMalloc((void **)&b_d, size);
    error_check(cerr, 1);

    cerr = cudaMalloc((void **)&c_d, size);
    error_check(cerr, 2);

    // Copy the inputs vectors to the device input
    cerr = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    error_check(cerr, 3);
    cerr = cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    error_check(cerr, 4);

    // Calcualte the thread block and grid sizes
    int threads_per_block = 256;
    // -> calculate the number of blocks in the grid
    printf("Threads per block: %d\nBlocks per grid: %d\n",
           threads_per_block,
           blocks_per_grid);

    // Launch the Vector Add Kernel
    vector_add<<<blocks_per_grid, threads_per_block>>>(n, a_d, b_d, c_d);

    // Copy result to host
    // -> Add device to host copy
    error_check(cerr, 5);

    // Verify the result
    for (int i = 0; i < n; ++i) {
        if (fabs(a_h[i] + b_h[i] - c_h[i]) > 1e-5) {
            printf("a: %f b: %f c:%f\n", a_h[i], b_h[i], c_h[i]);
            printf("Result incorrect in element: %d\n", i);
            exit(-1);
        }
    }

    printf("Test passed\n");

    // Free device memory
    cerr = cudaFree(a_d);
    error_check(cerr, 6);

    cerr = cudaFree(b_d);
    error_check(cerr, 7);

    cerr = cudaFree(c_d);
    error_check(cerr, 8);

    // Free host memory
    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}
