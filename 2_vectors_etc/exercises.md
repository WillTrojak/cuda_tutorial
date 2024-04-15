# Vectors et al. - 2x32 attention

In this exercise we'll explore vector types and some cahche hinting via
attention. We'll be working on an attention layer that has a a token size of 2
and a context window of size 32. (Oooo some lovely easier numbers to work with).
We will be doing this in a simple manner by using one thread per query and one
warp per context.

We're doing to ignore the sqrt(d) term that often comes in.

# Exercise 0

We've already got some of the starting point of the attention layer. We'll mae
use of sharfed memory again in a way very similar to the `1_shared` tasks.

* Start by adding the value and query parts.
    * Remeber the key and value parts are for the whole context, whereas the
      query is local. Ie we are dealing with one query per thread
* Next calculate the query-wise softmax.
* Multiple the softamx byt the value and write out

# Exercise 1

We will now try to make use of the built-in vector types.

* Try loading the `x` values used by thread into a `float2`
* The weights actually all fit in a `float4`, try loading the weights into 3
  `flaot4`
* Replace where the weights and `x` are used with the vector values

# Exercise 2

The loads and stores that we do to global memroy can probably be hinted. Check
out the options available at these 2 links [CUDA
Reference](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#load-functions-using-cache-hints)
anbd [PTX hint
definition](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)

* What cache hint can we do for the loads?
* What cahce hinting can we do for the store?
