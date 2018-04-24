"""
Module implementing FASTA:

An fast implementation of Proximal Gradient Descent solver
Created on Sun Mar 11 16:05:29 2018

@author: Proloy Das

@licence = apache 2.0
"""

import numpy as np
import tqdm
import time
from scipy import linalg


def fastapy(f, g, gradf, proxg, x0, beta=0.5, max_Iter=1000, tol=1e-8):
     """
     Solves
                        \min_{x} f(x) + g(x)
     given following inputs:

    :param f: function handle of smooth differentiable function,
    :math:                        $f(x)$

    :param g: function handle of non-smooth convex function, g(x)
    :math:                        $g(x)$

    :param gradf: function handle for gradient of smooth differentiable function
    :math:                     $\nabla f(x)$

    :param proxg: function handle for proximal operator of non-smooth convex function
    :math: ${\sf prox}_{\lambda g}(v) = \argmin_{x} g(x) + 1/(2*lambda)\| x-v\|**2

    :param x0: initial guess
            (M,) 1D array or (M,M) 2D array

    :param beta: backtracking parameter
            scalar float
            uses 0.5 by default.
    :param max_Iter: maximum number of iterations
            scalar integer
            uses 1000 by default
    :param tol: tolerance
            scalar float
            uses 1e-8 as default

    :return: a dict containing the
    "solution": (M,) 1D array or (M,M) 2D array
    "objective values": 1D array
    "residuals": 1D array

    Caution:
    Make sure that outputs of gradf and proxg is of same size as x.
    The implementation does not check for any such discrepancies.

    Usage: Solve following least square problem using fastapy
                min  .5||Ax-b||^2 + mu*|x|
    # Create function handles
    >>> def f(x): return 0.5 * linalg.norm(np.dot(A, x) - b, 2)**2  # f(x) = .5||Ax-b||^2
    >>> def gradf(x): return np.dot(A.T, np.dot(A, x) - b)  # gradient of f(x)
    >>> def g(x): return mu * linalg.norm(x, 1)  # mu|x|
    >>> def proxg(x, t): return shrink(x, mu*t)
    >>> def shrink(x, mu): return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0)) #proxg(z,t) = sign(x)*max(|x|-mu,0)
    # Call Solver
    >>> out = fastapy( f, g, gradf, proxg, x0, tol=1e-20)

    """
    
        # estimate Lipschitz constant ans initialize tau
    x = x0 + 0.01 * np.random.randn(x0.shape[0], x0.shape[1])
    L = np.square(linalg.norm(gradf(x0) - gradf(x), 'fro')) / np.square(linalg.norm(x0 - x, 'fro'))
    tau = 1 / L

    x = np.copy(x0)
    gradfx = gradf(x)

    # Save f(x) values for back-tracking

    fx = np.empty((0))
    fx = np.append(fx, f(x))

    # Save objective values for returing
    fval = np.empty((0))
    fval = np.append(fval, fx[-1] + g(x))

    # Save Residuals for returning
    residual = np.empty((0))

    for _ in tqdm.tqdm(range(max_Iter)):
        time.sleep(0.001)

        # backtracking for step size - tau
        z = proxg(x - tau * gradfx, tau)
        fk = np.max(fx)
        # fk = fx[-1]
        while f(z) > fk + np.sum(gradfx * (z - x)) + np.square(linalg.norm(z - x, 'fro')) / (2 * tau):
            tau = beta * tau
            z = proxg(x - tau * gradfx, tau)

        # Check for convergence and if reached break
        gradfz = gradf(z)
        residual = np.append(residual, linalg.norm(gradfz + (x - tau * gradfx - z) / tau, 'fro') ** 2)
        if residual[-1] / residual[0] < tol:
            break

        # choose next step size using adaptive BB method
        deltax = z - x
        deltaF = gradfz - gradfx
        n_deltax = linalg.norm(deltax, 'fro') ** 2
        n_deltaF = linalg.norm(deltaF, 'fro') ** 2
        innerproduct_xF = np.sum(deltax * deltaF)
        if n_deltax == 0:
            break
        elif (n_deltaF == 0) | (innerproduct_xF == 0):
            tau = 1 / L
        else:
            tau_s = n_deltax / innerproduct_xF  # steepest descent
            tau_m = innerproduct_xF / n_deltaF  # minimum residual
            # adaptive BB method
            if 2 * tau_m > tau_s:
                tau_k = tau_m
            else:
                tau_k = tau_s - 0.5 * tau_m
            if tau_k > 0:
                tau = tau_k

        x = np.copy(z)
        gradfx = np.copy(gradfz)
        fx = np.append(fx, f(x))
        fval = np.append(fval, fx[-1] + g(x))

    out = {"sol": x, "objective values": fval, "residual": residual}
    return out
