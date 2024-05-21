# Legendre polynomials with cuBLAS

This exercise will evaluate a large number of Legendre polynomials at a set of
points. To do this we can build a matrix and use cuBLAS (CUDA basic linear
algebra subprograms). We will use the level 3 function (matrix-matrix
functions). You can find the documentation of BLAS
[here](https://docs.nvidia.com/cuda/cublas/).

## Exercise 0
* Implement the recurrence relation for [Legendre
  polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials).
* Create a cuBLAS handle and set which stream it uses.
* Add a call to SGEMM

## Exercise 1
* Create multiple streams for the multiple mode matrices
* Create multiple cuBLAS handles and set there respective streams
* Add multiple calls to SGEMM

## Exercise 2
* Create a CUDA graph
* Record the activity of the single stream used to a graph
* Convert the recorded graph to an executable graph and run it
