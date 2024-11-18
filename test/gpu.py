# gpu_task.py
import cupy as cp

# Create arrays on the GPU
a = cp.array([1, 2, 3, 4, 5])
b = cp.array([10, 20, 30, 40, 50])

# Perform operations on the GPU
c = a * b
print("Result of GPU computation:", c)

# Transfer result back to CPU if needed
c_cpu = cp.asnumpy(c)
print("Transferred to CPU:", c_cpu)
