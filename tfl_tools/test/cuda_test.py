# created a cudatest environment on GPU for testing
from numba import cuda
import numpy
import math

@cuda.jit
def my_kernel(io_array):
    """
    Code for kernel.
    """
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # Check array boundaries
        io_array[pos] *= 2  # do the computation

def my_kernel_2D(io_array):
    x, y = cuda.grid(2)
    ### YOUR SOLUTION HERE
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.y
    #bwx = cuda.blockDim.x
    #bwy = cuda.blockDim.y
    io_array[tx,ty] *= 2  # do the computation

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
print('###################################')
data2 = numpy.ones((16, 16))
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
my_kernel_2D[blockspergrid, threadsperblock](data2)
print(data)
