# Device functions - approximating Gauss--Legendre points

## Starting exercise
* Start by implementing the function `bessel0_root`. This can be found in the
  paper linked. A copy of the paper is available in the box.
* Convert `olver_approx`, `tricomi_approx`, `bessel0_root` to device functions.
* Create a stream
* Add calls to `calculate_theta` and `invert_theta`

## Further exercise 1
We defined a number of constants for the `bessel0_root` function. Try to
explcicitly define device constants for these.

## Further exercise 2
* The pointer we use for x and theta nver alias, how can we use this to give
  further information to the compiler?
* Have a look at the bessel0_root function in solution_0. How similar is it to
  what you did? Can it be optimised (consider Horner's rule)?
* The device functions are fairly small, can we instruct the compiler to inline
  them?

## Further exercise 3
We defined a typedef and it is currently using doubles. What is the effect on
performance if we swap to single?
