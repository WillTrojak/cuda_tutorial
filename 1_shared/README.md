# Shared Memory - 32x32 linear layer and activation

We will be working on a weight matrix that is 32×32 and will have 'data' that
is 32 x `nbatch`.

## Exercise 0
* On the host, implement a linear layer with bias and ReLU activation.
  This will be used for testing and a base for the device code.
    * The main part of this is the matrix multiplication. Focus on that first.
* Implement the same thing for the GPU.
    * A good way to think about this is by considering a matrix multiplication
      as a series of dot-products.

## Exercise 1
Now we have a working device version, we will use shared memory.

* Declare some (_static_) shared memory for the weight and bias.
* Read the weight and bias into shared memory and use that in the matrix
  multiplication instead of the main memory version.
* Consider how you are loading the data into shared.
    * If two threads in a warp access shared memory and the pointer % 32 is the
      same then one will have to wait for the other. What can we do? Do you
      need to do anything?

Note: shared memory is allocated _per block_. Do you really want all the threads
participating?

## Exercise 2

Now instead of loading the weight and bias into shared, we are going to load
all the data, `x`, used in that block into shared.

* Add a `#define` to set the size of the block. This is because we are using
  static shared memory (rather than dynamic)
* Declare some (_static_) shared memory for the data, `x`, such that all the
  data values used by the block are loaded into shared.
* Update the dot-product loop to use the new shared version of `x`
* Is there anything we can do to the indexing to avoid bank conflicts?

## Exercise 2+

Now that we've written a few different approaches, let's benchmark them. The
file `shared_solution_012_time.cu` has all three solutions and measures the
time.
* You can play around this the batch size to see what happens.
* What do you notice?
* Why do you think this behaves the way it does?
