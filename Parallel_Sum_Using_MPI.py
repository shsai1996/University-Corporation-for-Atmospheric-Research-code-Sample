from numba import njit, prange
import numpy as np

# Numba's njit decorator with parallel=True enables OpenMP parallelism.
@njit(parallel=True)
def sum_of_squares(arr):
    result = 0
    # Using prange to parallelize the loop
    for i in prange(len(arr)):
        result += arr[i] ** 2
    return result

# Example usage
if __name__ == "__main__":
    # Create a large array of numbers
    array = np.arange(1, 10000001)
    
    # Calculate the sum of squares in parallel
    result = sum_of_squares(array)
    
    print("Sum of squares:", result)
