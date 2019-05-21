# created a cudatest environment on GPU for testing
from numba import cuda
import numpy

@cuda.jit
def my_kernel(io_array):
    """
    Code for kernel.
    """
    # code here

print(cuda.gpus)

# Create the data array - usually initialized some other way
data = numpy.ones(256)

# Set the number of threads in a block
threadsperblock = 32

# Calculate the number of thread blocks in the grid
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
my_kernel[blockspergrid, threadsperblock](data)

# Print the result
print(data)