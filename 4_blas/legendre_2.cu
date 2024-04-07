#include <stdio.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>


typedef float ftype;


void bstat_check(cublasStatus_t stat, int id) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed (%d) with error code %d\n", id, stat);
        exit(EXIT_FAILURE);
    }
}


void error_check(cudaError_t cerr, int id) {
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Failed (%d) with error code %s\n", id, cudaGetErrorString(cerr));
        exit(EXIT_FAILURE);
    }
}


void legendre(const int order, const float x, float * __restrict__ p, const int ldp) {
    if (order < 0) {
        fprintf(stderr, "legendre order should be zero or positive");
        exit(EXIT_FAILURE);
    }

    if (order >= 0)
        p[0] = 1.;

    if (order >= 1)
        p[ldp] = x;

    if (order > 1)
        for (int i = 2; i <= order; ++i)
            p[i*ldp] = ((2.*i - 1)*x*p[(i-1)*ldp] - (i - 1.)*p[(i-2)*ldp]) / i;
}


void projection_mat(const int n, const int order, const float *x,
                    float * __restrict__ mat) {
    for (int i = 0; i < n ; ++i){
        legendre(order, x[i], mat + i, n);
    }
}


int main(void) {
    cudaError_t cerr;
    cublasStatus_t bstat;

    int order = 2;
    int npnts = order + 1;
    int neles = 3;
    size_t size_proj = (order + 1) * npnts * sizeof(float);
    size_t size_pnts = npnts * sizeof(float);
    int nmode = neles * npnts;
    size_t size_mode = nmode * sizeof(float);

    // Allocate the host memory
    //   proj[npnts x order+1]
    //   mode[order+1 x neles]
    //   tran[npnts x neles]
    float *proj_h, *pnts_h, *tran_h;
    float *mode1_h, *mode2_h, *mode3_h;
    proj_h = (float *)malloc(size_proj);
    pnts_h = (float *)malloc(size_pnts);

    tran_h = (float *)malloc(3*size_mode);
    mode1_h = (float *)malloc(size_mode);
    mode2_h = (float *)malloc(size_mode);
    mode3_h = (float *)malloc(size_mode);

    // Initialise point vector
    for (int i = 0; i < npnts; ++i)
        pnts_h[i] = ((float)i) / ((float)npnts);

    // Initialise the mode values
    for (int i = 0; i < neles; ++i)
        for (int ii = 0; ii < npnts; ++ii) {
            mode1_h[i*npnts + ii] = rand() / (float)RAND_MAX;
            mode2_h[i*npnts + ii] = rand() / (float)RAND_MAX;
            mode3_h[i*npnts + ii] = rand() / (float)RAND_MAX;
        }

    // Initialise the projection matrix
    projection_mat(npnts, order, pnts_h, proj_h);

    // Allocate the device memory
    float *proj_d, *pnts_d, *tran_d;
    float *mode1_d, *mode2_d, *mode3_d;

    cerr = cudaMalloc((void **)&tran_d, 3*size_mode);
    error_check(cerr, 0);

    cerr = cudaMalloc((void **)&mode1_d, size_mode);
    error_check(cerr, 1);
    cerr = cudaMalloc((void **)&mode2_d, size_mode);
    error_check(cerr, 2);
    cerr = cudaMalloc((void **)&mode3_d, size_mode);
    error_check(cerr, 3);

    cerr = cudaMalloc((void **)&proj_d, size_proj);
    error_check(cerr, 4);

    cerr = cudaMalloc((void **)&pnts_d, size_pnts);
    error_check(cerr, 5);

    // Copy the matrices to the device
    cerr = cudaMemcpy(mode1_d, mode1_h, size_mode, cudaMemcpyHostToDevice);
    error_check(cerr, 6);
    cerr = cudaMemcpy(mode2_d, mode2_h, size_mode, cudaMemcpyHostToDevice);
    error_check(cerr, 7);
    cerr = cudaMemcpy(mode3_d, mode3_h, size_mode, cudaMemcpyHostToDevice);
    error_check(cerr, 8);
    cerr = cudaMemcpy(pnts_d, pnts_h, size_pnts, cudaMemcpyHostToDevice);
    error_check(cerr, 9);
    cerr = cudaMemcpy(proj_d, proj_h, size_proj, cudaMemcpyHostToDevice);
    error_check(cerr, 10);

    // Create a stream
    cudaStream_t stream1;
    cerr = cudaStreamCreate(&stream1);
    error_check(cerr, 11);

    // Create a cublas handle and set the stream
    cublasHandle_t handle1;
    bstat = cublasCreate(&handle1);
    bstat_check(bstat, 12);

    bstat = cublasSetStream(handle1, stream1);
    bstat_check(bstat, 13);

    // Create a cuda graph
    cudaGraph_t graph;
    // -> create a standard graph

    // Set the alpha and beta value for the gemm
    float alpha = 1., beta = 0.;

    // Start capturing the activity in stream1
    // -> begin capturing the activity in the stream

    // Call the gemm function
    // C = alpha * A * B + beta * C
    bstat = cublasSgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N,  npnts, neles,
                        order+1, &alpha, proj_d, npnts, mode1_d, order+1, &beta,
                        tran_d + 0*nmode, npnts);
    bstat = cublasSgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N,  npnts, neles,
                        order+1, &alpha, proj_d, npnts, mode2_d, order+1, &beta,
                        tran_d + 1*nmode, npnts);
    bstat = cublasSgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N,  npnts, neles,
                        order+1, &alpha, proj_d, npnts, mode3_d, order+1, &beta,
                        tran_d + 2*nmode, npnts);

    // Stop capturing the activity in stream1
    // -> end the stream capture and make a graph
    error_check(cerr, 14);

    // Make an executable graph
    // -> Instantiate an executable graph from the captured graph

    // Try running the graph
    // -> Run the executable graph
    error_check(cerr, 16);

    // Copy result to host
    cerr = cudaMemcpy(tran_h, tran_d, 3*size_mode, cudaMemcpyDeviceToHost);
    error_check(cerr, 17);

    // Destroy the handle
    bstat = cublasDestroy(handle1);
    bstat_check(bstat, 18);

    // Destroy execution graph
    cerr = cudaGraphDestroy(graph);
    error_check(cerr, 19);
    cerr = cudaGraphExecDestroy(graph_run);
    error_check(cerr, 20);

    // Free device memory
    cerr = cudaFree(proj_d);
    error_check(cerr, 21);

    cerr = cudaFree(pnts_d);
    error_check(cerr, 22);

    cerr = cudaFree(mode1_d);
    error_check(cerr, 23);
    cerr = cudaFree(mode2_d);
    error_check(cerr, 24);
    cerr = cudaFree(mode3_d);
    error_check(cerr, 25);

    cerr = cudaFree(tran_d);
    error_check(cerr, 26);

    // Free host memory
    free(pnts_h);
    free(proj_h);
    free(mode1_h);
    free(mode2_h);
    free(mode3_h);
    free(tran_h);

    return 0;
}
