# cuBLAS - legendre polynomials

This exercise will evaluate a large number of Legendre polynomials at a set of
points. TO do this we can build a matrix and then use cuBLAS (CUDA basic linear
algebra subprograms). We'll be using the level 3 function (matrix-matrix
functions). You can find the docmentation of blas
[here](https://docs.nvidia.com/cuda/cublas/).

## Exercise 0
* Implement the recurance relation for legendre polynomials. (See wikipedia)
* Create a cublas handle and set which stream it uses.
* Add a call to sgemm

## Exercise 1
* Create multiple streams for the multiple mode matrices
* Create multiple cublas handles and set there respective streams
* Add multiple calls to sgemm

## Exercise 2
* Create a cuda graph
* Record the activity of the single stream used to a graph
* Convert the recorded graph to an executable graph and run it
