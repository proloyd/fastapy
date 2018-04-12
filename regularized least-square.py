"""
This script shows how to use FASTA to solve regularized least-square problem:
        min  .5||Ax-b||^2 + mu*|x|
Where A is an MxN matrix, b is an Mx1 vector of measurements, and x is the Nx1 vector of unknowns.
The parameter 'mu' controls the strength of the regularizer.

"""

import numpy as np
from scipy import linalg
from FASTA import fastapy, PGD
import matplotlib.pyplot as plt


# Define problem parameters
M = 200  # number of measurements
N = 1000  # dimension of sparse signal
K = 10    # signal sparsity
mu = .02  #  regularization parameter
sigma = 0.01  #  The noise level in 'b'

print 'Testing sparse least-squares with N={:}, M={:}'.format(N, M)

# Create sparse signal
x = np.zeros((N, 1))
perm = np.random.permutation(N)
x[perm[0:K]] = 1


# define random Gaussian matrix
A = np.random.randn(M, N)
A = A/linalg.norm(A, 2)  # Normalize the matrix so that our value of 'mu' is fairly invariant to N

# Define observation vector
b = np.dot(A, x)
b = b + sigma * np.random.randn(*b.shape)  # add noise

#  The initial iterate:  a guess at the solution
x0 = np.zeros((N, 1))

# Create function handles
def f(x):
    """.5||Ax-b||^2"""
    return 0.5 * linalg.norm(np.dot(A, x) - b, 2)**2


def gradf(x):
    """gradient of f(x)"""
    return np.dot(A.T, np.dot(A, x) - b)


def g(x):
    "|x| < mu"
    return mu * linalg.norm(x, 1)


def proxg(x, t):
    """
    proximal operator for g(x)
    proxg(z,t) = argmin t*mu*|x|+.5||x-z||^2
    """
    return shrink(x, mu*t)


def shrink(x, mu):
    """
    Soft theresholding function
    mu = threshold
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


# Call Solver
[solution, fval, residual] = fastapy( f,g,gradf,proxg,x0 )

# plot results
plt.figure('sparse least-square')
plt.subplot(2, 1, 1)
plt.stem(x,  linefmt=':')
plt.stem(solution)
plt.xlabel('Index')
plt.ylabel('Signal Value')

plt.subplot(2, 1, 2)
plt.semilogy(residual)
