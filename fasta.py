"""
python implementation of FASTA:

An fast implementation of Proximal Gradient Descent solver
Created on Sun Mar 11 16:05:29 2018

@author: Proloy Das
"""

import numpy as np
import tqdm
import time
from scipy import linalg


def fastapy(f, g, gradf, proxg, x0, beta=0.5, max_Iter=1000, tol=1e-8):
    """

    :param f: function handle of smooth differentiable function
    :param g: function handle of non-smooth convex function
    :param gradf: function handle for gradient of smooth differentiable function
    :param proxg: function handle for proximal operator of non-smooth convex function
    :param x0: initial guess
    :param beta: parameter
    :param max_Iter: maximum number of iteration
    :param tol: tolerance

    :return: solution, function values, residual values
    """

    # estimate Lipschitz constant ans initialize tau
    x = x0 + 0.001 * np.random.randn (x0.shape[0], x0.shape[1])
    L = linalg.norm(gradf(x0) - gradf(x), 'fro')**2 / linalg.norm(x0 - x, 'fro')**2
    tau = 1 / L

    x = np.copy(x0)
    fval = np.empty(0)
    fx = np.empty(0)
    residual = np.empty(0)
    for _ in tqdm.tqdm(range(max_Iter)):
        time.sleep(0.001)
        fx = np.append(fx, f(x))

        # backtracking line search
        grad = gradf(x)
        z = proxg(x - tau * grad, tau)
        # fk = np.max(fx)
        fk = fx[-1]
        while f(z) > fk + np.sum(grad * (z - x)) + linalg.norm(z - x, 'fro')**2 / (2 * tau):
            tau = beta * tau
            z = proxg(x - tau * grad, tau)

        # compute residual
        residual = np.append(residual, linalg.norm(gradf(z) + (x - tau * grad - z) / tau, 'fro') ** 2)

        x = np.copy(z)
        fval = np.append(fval, f(x) + g(x))

        # stopping condition
        if residual[-1] / residual[0] < tol:
            break

        # choose next step size using adaptive BB method
        deltax = z - x
        deltaF = gradf(z) - gradf(x)
        tau_s = linalg.norm(deltax, 'fro')**2 / np.sum(deltax * deltaF)  # steepest descent
        tau_m = np.sum(deltax * deltaF) / linalg.norm(deltaF, 'fro')**2  # minimum residual
        if 2 * tau_m > tau_s:
            tau_k = np.copy(tau_m)
        else:
            tau_k = tau_s - 0.5 * tau_m
        if tau_k > 0:
            tau = np.copy(tau_k)

    return x, fval, residual
