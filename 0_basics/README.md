# In the beginning - adding vectors

To get us going we are going to try a simple kernel that reads in two vectors,
`a` and `b`, adds them, and returns the result in a third vector, `c`.

* Start by adding the call that copies the device pointer `c_d` to the host
  pointer `c_h`.
* Using a simple scheme were each thread does a single addition, define the
  number of blocks given that the vectors have `n` entries and the threads per
  block is already set.
* Define the index that a given thread is working on.
* Perform the calculation.
* You'll probably have more threads than needed what can we do about that?
