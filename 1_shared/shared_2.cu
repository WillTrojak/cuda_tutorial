#include <stdio.h>
#include <algorithm>

#include <cuda_runtime.h>

typedef float ftype;

// -> define a CPP macro for the block size
#define

void error_check(cudaError_t cerr, int id) {
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Failed (%d) with error code %s\n", id, cudaGetErrorString(cerr));
        exit(EXIT_FAILURE);
    }
}


// z[n * m] = sigma(w[n * k] * x[k * m] + b[n])
void layer_test(int n, int m, int k, const ftype *w, const ftype *x,
                const ftype *b, ftype *z) {
    for (int i=0; i<m; ++i) {
        for (int ii=0; ii<n; ++ii) {
            ftype acc = b[ii];
            for (int iii=0; iii<k; ++iii) {
                acc += w[ii + iii*n]*x[iii + i*k];
            }
            z[ii + i*k] = std::max((ftype)0., acc);

        }
    }
}


// z = sigma(w*x + b)
__global__ void layer_32x32_shared_x(const int n, const ftype __restrict__ *w,
                                     const ftype __restrict__ *x,
                                     const ftype __restrict__ *b,
                                     ftype __restrict__ *z) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int batch = k / 32;
    int warp = threadIdx.x / 32;

    // -> Calculate the number of batches that a whole block calculates
    // -> Calculate the block local batch number, ie of the n batches a block
    //    does which one is this thread working on.

    int row = threadIdx.x % warpSize;
    // -> Declare the shared memory of the x this block will work on. You'll
    //    need to use your CPP defined block size.

    if (k < n) {
        // -> Load data into shared memory

        __syncthreads();

        // Initialise an accumulator with the bias
        ftype acc = b[row];
        for (int i=0; i < 32; ++i)
        {
            // -> update the dot-product accumulation to use shared memory
        }

        // Apply ReLU activation and write to z
        z[row + 32*batch] = max((ftype)0., acc);
    }
}


int main(void) {
    cudaError_t cerr;

    const int nbatch = 16;
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
    // -> Update the block size to use your CPP value
    int threads_per_block = ;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    printf("Threads per block: %d\nBlocks per grid: %d\n",
           threads_per_block,
           blocks_per_grid);

    // Launch the kernel
    layer_32x32_shared_x<<<blocks_per_grid, threads_per_block>>>(n, w_d, x_d, b_d, z_d);

    // Copy result to host
    cerr = cudaMemcpy(z_h, z_d, size_z, cudaMemcpyDeviceToHost);
    error_check(cerr, 7);

    // Verify the result
    layer_test(nfeat, nbatch, nfeat, w_h, x_h, b_h, z_t);
    for (int i=0; i < nbatch; ++i)
        for (int ii=0; ii < nfeat; ++ii) {
            ftype _z_h = z_h[ii + i*nfeat];
            ftype _z_t = z_t[ii + i*nfeat];
            if (fabs(_z_h - _z_t) > 1e-4) {
                printf("diff: %f\n", fabs(_z_h - _z_t));
                printf("value: %f test: %f\n", _z_h, _z_t);
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
