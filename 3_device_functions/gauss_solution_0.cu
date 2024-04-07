#include <stdio.h>

#include <cuda_runtime.h>

typedef double ftype;

__device__ ftype tricomi_approx(const int n, const int k);
__device__ ftype olver_approx(const int n, const int k);
__device__ ftype bessel0_root(const int k);

void error_check(cudaError_t cerr, int id) {
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Failed (%d) with error code %s\n", id, cudaGetErrorString(cerr));
        exit(EXIT_FAILURE);
    }
}


__global__ void invert_theta(int n, const ftype *theta, ftype *x) {
    int k = blockDim.x*blockIdx.x + threadIdx.x;
    int n2 = floor(n*0.5);

    if (k < n2) {
        ftype cos_theta = cos(theta[k]);
        x[k] = cos_theta;
        x[n - k - 1] = -cos_theta;
    }
}

__global__ void calculate_theta(int n, ftype *theta) {
    int k = blockDim.x*blockIdx.x + threadIdx.x;
    int n2 = floor(n*0.5);

    if (k < n2) {
        ftype x_k = tricomi_approx(n, k+1);
        if(fabs(x_k) > 0.5) x_k = olver_approx(n, k+1);

        theta[k] = acos(x_k);
    }

}

__device__ ftype tricomi_approx(const int n, const int k) { // best for |x| < 0.5
    const ftype pi = 3.141592653589793;

    ftype phi = (k - 0.25)*pi / (n + 0.5);

    ftype x_k = 39 - 28/(pow(sin(phi), 2));
    x_k /= -(384 * pow(n, 4));
    x_k -= (n - 1.) / (8 * pow(n, 3));
    x_k += 1;
    x_k *= cos(phi);

    return x_k;
}

__device__ ftype olver_approx(const int n, const int k) { // best for 0.5 <= |x| <=1
    ftype nph = n + 0.5;
    ftype psi = bessel0_root(k) / nph;

    ftype x_k = psi / tan(psi) - 1;
    x_k /= (8 * psi * nph * nph);
    x_k += psi;
    x_k = cos(x_k);

    return x_k;
}

__device__ ftype bessel0_root(const int k) {
    const ftype pi = 3.141592653589793;
    // Approximation parameters from JCP 42, pp. 403-405 (1981)
    const ftype a0 = 0.0682894897349453;
    const ftype a1 = 0.131420807470708;
    const ftype a2 = 0.0245988241803681;
    const ftype a3 = 0.000813005721543268;

    const ftype b0 = 1.;
    const ftype b1 = 1.16837242570470;
    const ftype b2 = 0.200991122197811;
    const ftype b3 = 0.00650404577261471;

    ftype beta = (k - 0.25) * pi;

    ftype j0 = a0 + a1*powf(beta, 2) + a2*powf(beta, 4) + a3*powf(beta, 6);
    j0 /= (beta*(b0 + b1*powf(beta, 2) + b2*powf(beta, 4) + b3*powf(beta, 6)));
    j0 += beta;

    return j0;
}

int main(void) {
    cudaError_t cerr;

    int n = 11;
    size_t size = n * sizeof(ftype);

    // Allocate the host memory
    ftype *theta_h = (ftype *)malloc(size);
    ftype *x_h = (ftype *)malloc(size);

    // Allocate the device memory
    ftype *theta_d, *x_d;
    cerr = cudaMalloc((void **)&theta_d, size);
    error_check(cerr, 0);
    cerr = cudaMalloc((void **)&x_d, size);
    error_check(cerr, 1);

    // Create the stream
    cudaStream_t stream;
    cerr = cudaStreamCreate(&stream);
    error_check(cerr, 2);

    // Calculate the thread size
    int threads_per_block = 256;
    int nh = floor(n*0.5);
    int blocks_per_grid = (nh + threads_per_block - 1) / threads_per_block;
    printf("Threads per block: %d\nBlocks per grid: %d\n",
           threads_per_block,
           blocks_per_grid);

    // Launch the CUDA Kernels
    calculate_theta<<<blocks_per_grid, threads_per_block, 0, stream>>>(n, theta_d);
    invert_theta<<<blocks_per_grid, threads_per_block, 0, stream>>>(n, theta_d, x_d);

    // Copy result to host
    cerr = cudaMemcpy(theta_h, theta_d, size, cudaMemcpyDeviceToHost);
    error_check(cerr, 3);
    cerr = cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
    error_check(cerr, 4);

    // Print the result
    if (n < 20) {
        for (int i = 0; i < n; ++i) {
            printf("x_h[%d]: %f\n", i, x_h[i]);
        }
    }

    // Free device memory
    cerr = cudaFree(theta_d);
    error_check(cerr, 5);

    cerr = cudaFree(x_d);
    error_check(cerr, 6);

    // Free host memory
    free(theta_h);
    free(x_h);

    return 0;
}
