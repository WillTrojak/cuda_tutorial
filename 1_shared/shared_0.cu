#include <algorithm>
#include <stdio.h>

#include <cuda_runtime.h>

typedef float ftype;

void error_check(cudaError_t cerr, int id) {
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Failed (%d) with error code %s\n", id, cudaGetErrorString(cerr));
        exit(EXIT_FAILURE);
    }
}


// z[n * m] = sigma(w[n * k] * x[k * m] + b[n])
// -> add the linear layer with bias and ReLU activation. I've already given all
//    the for loops you'll need.
//    Note: std::max
void layer_test(int n, int m, int k, const ftype *w, const ftype *x,
                const ftype *b, ftype *z) {
    for (int i=0; i<m; ++i) {
        for (int ii=0; ii<n; ++ii) {
            for (int iii=0; iii<k; ++iii) {

            }
        }
    }
}


// z = sigma(w*x + b)
__global__ void layer_32x32(const int n, const ftype __restrict__ *w,
                            const ftype __restrict__ *x,
                            const ftype __restrict__ *b,
                            ftype __restrict__ *z) {
    // -> Calculate your global thread index, j
    // -> Calcualte the batch and row.

    if (j < n) {
        // -> Initialise an accumulator with the bias

        // -> get each thread to perform one of the dot-producs

        // -> ReLU
        // -> Set z
    }
}


int main(void) {
    cudaError_t cerr;

    // Some constants and sizes
    const int nbatch = 1024;
    const int nfeat = 32;
    const int n = nbatch * nfeat;
    size_t size_w = nfeat * nfeat * sizeof(ftype);
    size_t size_b = nfeat * sizeof(ftype);
    size_t size_x = n * sizeof(ftype);
    size_t size_z = n * sizeof(ftype);

    // Allocate the host memory
    ftype *w_h, *b_h, *x_h, *z_h, *z_t;

    w_h = (ftype *)malloc(size_w);
    b_h = (ftype *)malloc(size_b);
    x_h = (ftype *)malloc(size_x);
    z_h = (ftype *)malloc(size_z);
    z_t = (ftype *)malloc(size_z);

    // Initialise the weights and biases on the host
    for (int i = 0; i < nfeat; ++i) {
        b_h[i] = 2*(rand() / (ftype)RAND_MAX) - 1;
        for (int ii = 0; ii < nfeat; ++ii)
            w_h[ii + nfeat*i] = 2*(rand() / (ftype)RAND_MAX) - 1;
    }

    // Initialise the data
    for (int i = 0; i < nbatch; ++i)
        for (int ii = 0; ii < nfeat; ++ii)
            x_h[ii + nfeat*i] = 2*(rand() / (ftype)RAND_MAX) - 1;

    // Allocate the device vectors
    ftype *w_d, *b_d, *x_d, *z_d;
    cerr = cudaMalloc((void **)&w_d, size_w);
    error_check(cerr, 0);

    cerr = cudaMalloc((void **)&b_d, size_b);
    error_check(cerr, 1);

    cerr = cudaMalloc((void **)&x_d, size_x);
    error_check(cerr, 2);

    cerr = cudaMalloc((void **)&z_d, size_z);
    error_check(cerr, 3);

    // Copy the weights, biases, and data to the host
    cerr = cudaMemcpy(w_d, w_h, size_w, cudaMemcpyHostToDevice);
    error_check(cerr, 4);
    cerr = cudaMemcpy(b_d, b_h, size_b, cudaMemcpyHostToDevice);
    error_check(cerr, 5);
    cerr = cudaMemcpy(x_d, x_h, size_x, cudaMemcpyHostToDevice);
    error_check(cerr, 6);

    // Calcualte the thread block and grid sizes
    int threads_per_block = 128;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    printf("Threads per block: %d\nBlocks per grid: %d\n",
           threads_per_block,
           blocks_per_grid);

    // Launch the kernel
    // -> Add launch of DEVICE kernel

    // Copy result to host
    cerr = cudaMemcpy(z_h, z_d, size_z, cudaMemcpyDeviceToHost);
    error_check(cerr, 7);

    // Verify the result
    // -> Add launch of the HOST version so we can test
    for (int i=0; i < nbatch; ++i)
        for (int ii=0; ii < nfeat; ++ii) {
            if (fabs(z_h[ii + i*nfeat] - z_t[ii + i*nfeat]) > 1e-5) {
                printf("value: %f test: %f\n", z_h[i], z_t[i]);
                printf("Result incorrect in element: (%d %d)\n", ii, i);
                exit(-1);
            }
        }

    printf("Test passed\n");

    // Free device memory
    cerr = cudaFree(w_d);
    error_check(cerr, 6);

    cerr = cudaFree(b_d);
    error_check(cerr, 7);

    cerr = cudaFree(x_d);
    error_check(cerr, 8);

    cerr = cudaFree(z_d);
    error_check(cerr, 9);

    // Free host memory
    free(w_h);
    free(b_h);
    free(x_h);
    free(z_h);
    free(z_t);

    return 0;
}
