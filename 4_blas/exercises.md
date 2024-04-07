# Exercise 0
* Implement the recurance relation for legendre polynomials. (See wikipedia)
* Create a cublas handle and set which stream it uses.
* Add a call to sgemm

# Exercise 1
* Create multiple streams for the multiple mode matrices
* Create multiple cublas handles and set there respective streams
* Add multiple calls to sgemm

# Exercise 2
* Create a cuda graph
* Record the activity of the single stream used to a graph
* Convert the recorded graph to an executable graph and run it
