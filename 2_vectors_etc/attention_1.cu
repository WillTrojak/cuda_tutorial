#include <algorithm>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>

typedef float ftype;
typedef float2 vtype2;
typedef float4 vtype4;

#define BLOCKDIMX 128

void error_check(cudaError_t cerr, int id) {
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Failed (%d) with error code %s\n", id, cudaGetErrorString(cerr));
        exit(EXIT_FAILURE);
    }
}


// x [n_token * n_context * n_batch]
// wq [n_token * n_token]
// wk [n_token * n_token]
// wv [n_token * n_token]
// z [n_token * n_context * n_batch]
__global__ void attention_2x32(const int n,
                              const ftype __restrict__ *wq,
                              const ftype __restrict__ *wk,
                              const ftype __restrict__ *wv,
                              const ftype __restrict__ *x,
                              ftype __restrict__ *z) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int batch = j / 32;
    int l_batch = threadIdx.x / 32;
    int bld = l_batch*64;
    int i = threadIdx.x % warpSize;

    ftype __shared__ k_s[(BLOCKDIMX/32)*32*2], v_s[(BLOCKDIMX/32)*32*2];

    if (j < n) {
        // -> Load x into a float2
        // -> Load wq into a float4
        // -> Load wk into a float4
        // -> Load wv into a float4

        // Calculate key
        // -> replace the use of x and wk with the vector versions
        k_s[i + 0*32 + bld] =
        k_s[i + 1*32 + bld] =

        // Calculate value
        // -> replace the use of x and wv with the vector versions
        v_s[i + 0*32 + bld] =
        v_s[i + 1*32 + bld] =

        // Calculate query
        ftype q[2];
        // -> replace the use of x and wq with the vector versions
        q[0] =
        q[1] =

        __syncthreads();

        // Softmax
        ftype qkt[32], denom = 0.;
        for (int ii=0; ii<32; ++ii) {
            qkt[ii] = expf(q[0]*k_s[ii + 0*32 + bld] + q[1]*k_s[ii + 1*32 + bld]);
            denom += qkt[ii];
        }
        ftype rcp_denom = 1. / denom; // Why did I do this?? What is the benefit

        // Apply denom and value
        for (int iii=0; iii<2; ++iii) {
            ftype v = 0.;
            for (int ii=0; ii<32; ++ii) {
                v += v_s[ii + iii*32 + bld] * qkt[ii];
            }
            z[iii + 2*i + 64*batch] = v * rcp_denom;
        }
    }
}


int main(void) {
    cudaError_t cerr;

    // Some constants and sizes
    const int nbatch = 1024;
    const int ncontext = 32;
    const int ntoken = 2;
    const int n = nbatch * ncontext * ntoken;
    size_t size_w = ntoken * ntoken * sizeof(ftype);
    size_t size_x = n * sizeof(ftype);
    size_t size_z = n * sizeof(ftype);

    // Allocate the host memory
    ftype *wq_h, *wk_h, *wv_h, *x_h, *z_h, *z_t;

    wq_h = (ftype *)malloc(size_w);
    wk_h = (ftype *)malloc(size_w);
    wv_h = (ftype *)malloc(size_w);
    x_h = (ftype *)malloc(size_x);
    z_h = (ftype *)malloc(size_z);
    z_t = (ftype *)malloc(size_z);

    // Initialise the weights on the host
    for (int i = 0; i < ntoken; ++i) {
        for (int ii = 0; ii < ntoken; ++ii) {
            wq_h[ii + ntoken*i] = 2*(rand() / (ftype)RAND_MAX) - 1;
            wk_h[ii + ntoken*i] = 2*(rand() / (ftype)RAND_MAX) - 1;
            wv_h[ii + ntoken*i] = 2*(rand() / (ftype)RAND_MAX) - 1;
        }
    }

    // Initialise the data
    for (int i = 0; i < n; ++i)
            x_h[i] = 2*(rand() / (ftype)RAND_MAX) - 1;

    // Allocate the device memory
    ftype *wq_d, *wk_d, *wv_d, *x_d, *z_d;
    cerr = cudaMalloc((void **)&wq_d, size_w);
    error_check(cerr, 0);

    cerr = cudaMalloc((void **)&wk_d, size_w);
    error_check(cerr, 1);

    cerr = cudaMalloc((void **)&wv_d, size_w);
    error_check(cerr, 2);

    cerr = cudaMalloc((void **)&x_d, size_x);
    error_check(cerr, 3);

    cerr = cudaMalloc((void **)&z_d, size_z);
    error_check(cerr, 4);

    // Copy the weights, and data to the host
    cerr = cudaMemcpy(wq_d, wq_h, size_w, cudaMemcpyHostToDevice);
    error_check(cerr, 5);
    cerr = cudaMemcpy(wk_d, wk_h, size_w, cudaMemcpyHostToDevice);
    error_check(cerr, 6);
    cerr = cudaMemcpy(wv_d, wv_h, size_w, cudaMemcpyHostToDevice);
    error_check(cerr, 7);
    cerr = cudaMemcpy(x_d, x_h, size_x, cudaMemcpyHostToDevice);
    error_check(cerr, 8);
    cerr = cudaMemcpy(z_d, z_h, size_z, cudaMemcpyHostToDevice);
    error_check(cerr, 9);

    // Calcualte the thread block and grid sizes
    int threads_per_block = BLOCKDIMX;
    int batch_per_block = (BLOCKDIMX  + 31) / 32;
    int blocks_per_grid = (nbatch + batch_per_block - 1) / batch_per_block;
    printf("Threads per block: %d\nBlocks per grid: %d\n",
           threads_per_block,
           blocks_per_grid);

    // Launch the kernel
    attention_2x32<<<blocks_per_grid, threads_per_block>>>(n, wq_d, wk_d, wv_d, x_d, z_d);

    // Copy result to host
    cerr = cudaMemcpy(z_h, z_d, size_z, cudaMemcpyDeviceToHost);
    error_check(cerr, 10);

    // Free device memory
    cerr = cudaFree(wq_d);
    error_check(cerr, 11);

    cerr = cudaFree(wk_d);
    error_check(cerr, 12);

    cerr = cudaFree(wv_d);
    error_check(cerr, 13);

    cerr = cudaFree(x_d);
    error_check(cerr, 14);

    cerr = cudaFree(z_d);
    error_check(cerr, 15);

    // Free host memory
    free(wq_h);
    free(wk_h);
    free(wv_h);
    free(x_h);
    free(z_h);
    free(z_t);

    return 0;
}
