from mpi4py import MPI
import numpy as np

def parallel_sum(data):
    # Get the rank (process ID) and size (total number of processes)
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # Determine the portion of the array to be handled by each process
    local_n = len(data) // size
    start = rank * local_n
    end = start + local_n

    # Each process calculates the sum of its portion
    local_sum = np.sum(data[start:end])

    # Gather all local sums to the root process (rank 0)
    global_sum = MPI.COMM_WORLD.reduce(local_sum, op=MPI.SUM, root=0)

    return global_sum

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define the size of the array
    n = 1000000

    if rank == 0:
        # Only the root process initializes the full array
        data = np.random.rand(n)
    else:
        data = None

    # Broadcast the array to all processes
    data = comm.bcast(data, root=0)

    # Compute the parallel sum
    total_sum = parallel_sum(data)

    # The root process prints the final sum
    if rank == 0:
        print(f"Total Sum: {total_sum}")

if __name__ == "__main__":
    main()
