import numpy as np
# import sys
# from scipy.linalg import eigh
# import warnings
import mpi4py as mpi
from functools import reduce
from operator import mul
from mpi4py import MPI
from scipy.linalg import lapack
# import scipy.linalg.blas as blas


def prod(iterable):
    """
    Simple product function.
    
    Parameters
    ----------
    iterable : An iterable with things that can be multiplied
               together.
    
    Returns
    -------
    value : The product of the values.
    
    """
    return reduce(mul, iterable, 1)



def mpidot(comm, A, B):
    rank = comm.Get_rank()
    size = comm.Get_size()
    # if A == None:
        # print(rank)
    # else:
    mat1 = A.copy()
    mat2 = B.copy()
    
    N,M = mat1.shape
    K = mat2.shape[1]
    assert mat1.shape[1] == mat2.shape[0]

    rows_per_process = N // size
    sendbuf_A = np.zeros((rows_per_process, M))
    recvbuf_C = np.zeros((N, K))

    comm.Scatterv(mat1, sendbuf_A, root=0)

    comm.Bcast(mat2, root=0)

    local_C = np.dot(sendbuf_A, mat2)

    comm.Allgatherv(local_C, recvbuf_C)
    return recvbuf_C
                         

def mpitensordot(comm, tensor1, tensor2, axes):
    """
    Contracts two tensors together according to `axes' using mpi.

    Parameters
    ----------
    tensor1            : The first tensor to contract.
    tensor2            : The second tensor that's contracted.
    axes : The indices over which the two tensors will be
                         contracted.

    Returns
    -------
    new_array : A new tensor built by contracting tensor2 and this tensor
                over their common indices.
                    
    """
    assert (len(axes) == 2)
    
    # get the tensor shapes
    ts1 = tensor1.shape
    ts2 = tensor2.shape
    
    # get the contracted indices
    ax1 = axes[0] # this is a tuple of indices for tensor1
    ax2 = axes[1] # ditto for tensor2
    
    # build the transposed tuples
    idx1 = list(range(len(ts1)))
    idx2 = list(range(len(ts2)))
    for n in ax1:
        idx1.remove(n)
    for n in ax2:
        idx2.remove(n)
    id1f = tuple(list(idx1) + list(ax1))
    id2f = tuple(list(ax2) + list(idx2))
    
    # transpose the input tensors to prepare
    # for matrix multiplication
    tleft = tensor1.transpose(id1f)
    tright = tensor2.transpose(id2f)
    
    # now sperate and reshape into two-index objects
    ts1 = tleft.shape
    ts2 = tright.shape
    left = ts1[:len(idx1)]
    right = ts2[len(ax2):]
    final = tuple(list(left) + list(right))
    assert (len(ax2) == (len(ts1)-len(idx1)))
    tleft = tleft.reshape((prod(ts1[:len(idx1)]), prod(ts1[len(idx1):])))
    tright = tright.reshape((prod(ts2[:len(ax2)]), prod(ts2[len(ax2):])))

    tleft = mpidot(comm, tleft, tright)
    # print(tleft)
    tleft = tleft.reshape(final)
    return tleft


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define matrix dimensions
n = 100
m = 50
k = min(m, n)

if rank == 0:
    # Initialize matrix A
    A = np.random.rand(m, n)

# Scatter matrix A to different processes
local_A = np.empty((m // size, n))
comm.Scatter(A, local_A, root=0)

# Perform parallel SVD using ScaLAPACK
u, s, vt, good = lapack.dgesvd(local_A, full_matrices=False)

# Gather the results of SVD
global_u = np.zeros((m, k))
global_s = np.zeros(k)
global_vt = np.zeros((k, n))

comm.Gather(u, global_u, root=0)
comm.Gather(s, global_s, root=0)
comm.Gather(vt, global_vt, root=0)

if rank == 0:
    # Combine singular values and left singular vectors
    U = np.zeros((m, k))
    Sigma = np.zeros((k, k))
    VT = np.zeros((k, n))
    U[:m, :k] = global_u
    np.fill_diagonal(Sigma, global_s)
    VT[:k, :n] = global_vt

    # # Combine singular values and left singular vectors
    # U = np.hstack(global_u)
    # Sigma = np.diag(global_s)
    # VT = global_vt

    # Combine local results to get full U, Sigma, VT matrices
    # U_full = np.zeros((m, m))
    # Sigma_full = np.zeros((m, n))
    # VT_full = np.zeros((n, n))
    # U_full[:, :k] = U
    # Sigma_full[:k, :k] = Sigma
    # VT_full[:k, :] = VT

    # Reconstruct the original matrix A from U, Sigma, VT
    A_reconstructed = np.dot(U, np.dot(Sigma, VT))
    print("Original matrix A")
    print(A)
    print("Reconstructed matrix A")
    print(A_reconstructed)

    # # Reconstruct the original matrix A from U, Sigma, VT
    # A_reconstructed = np.dot(U_full, np.dot(Sigma_full, VT_full))
    # print("Original matrix A")
    # print(A)
    # print("Reconstructed matrix A")
    # print(A_reconstructed)

