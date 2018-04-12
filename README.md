# fastapy
Python implementation of fast adaptive proximal gradient descent algorithm

Proximal gradient descent (also known as forward backward splitting or FBS) 
method is a way to solve high-dimensional optimization problems of form: 

                     minimize f(x) + g(x)

where $f(x)$ is convex and differentiable non-differentiable but $g(x)$ is 
typically not smooth but convex function. It is a two stage method which 
addresses two terms in the abovementioned problem separately.

At $l$th iteration of the algorithm, we have:

            x_{l+1} = proxg(x_l - eta_l*gradf, eta_l)

where eta_l is the (variable) step-size at lth iteration; and proxg, 
gradf denotes proximal operator of g(x) and gradient operator of f(x) 
respectively.

The iterations converges to a fixed point given that 

     0  <  eta_l  <  L(gradf)

where L(gradf) is  Lipschitz constant of gradf. Since L(gradf) for the 
problem is not known, back-tracking line-search is used to guarantee the 
stability of the algorithm. Along with that, in order to accelerate the 
convergence, we initialized the step-sizes using "adaptive" BB-method. 
The implementation is largely inspired by FASTA solver[1].

The assiciated script 'regularized least-square.py'shows how to use fastapy to solve 
regularized least-square problems.

[1] Goldstein, Tom, Christoph Studer, and Richard Baraniuk. "A field guide to forward-
backward splitting with a FASTA implementation." arXiv preprint arXiv:1411.3406 (2014).
