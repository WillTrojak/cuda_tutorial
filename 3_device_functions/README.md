# Device functions - approximating Gauss--Legendre points

In this exercise we will approximate the position of Gauss--Legendre quadrature
points. For this it will be useful to use device functions, i.e. a function on
the device called from on the device.

## Exercise 0
* Start by implementing the function `bessel0_root`. This can be found in the
  paper linked. A copy of the paper is available in the box.
* Convert `olver_approx`, `tricomi_approx`, `bessel0_root` to device functions.
* Update the prototypes for those functions too.
* Create a stream
* Add calls to `calculate_theta` and `invert_theta`

## Exercise 1
* We defined a number of constants for the `bessel0_root` function. Try to
  explcicitly define device `__constant__` for these.

## Exercise 2
* The pointer we use for x and theta never alias, how can we use this to give
  further information to the compiler?
* Have a look at the bessel0_root function in solution_0. How similar is it to
  what you did? Can it be optimised (consider Horner's rule)?
* The device functions are fairly small, can we instruct the compiler to inline
  them?

## Exercise 3

We created a type alias (`typedef`) as double precision (`double`). What is the
effect on performance if we swap to single precision (`float`)?
