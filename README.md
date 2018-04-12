# fastapy
Python implementation of fast adaptive proximal gradient descent algorithm

Proximal gradient descent (also known as forward backward splitting or FBS) 
method is a way to solve high-dimensional optimization problems of form: 

\begin{align} 
\min_{x \in \mathbbm{R}^n} f(x) + g(x)
\end{align}

where $f(x)$ is convex and differentiable non-differentiable but $g(x)$ is 
typically not smooth but convex function. It is a two stage method which 
addresses two terms in \eqref{eq:PGD1} separately.

At $l$th iteration of the algorithm, we have:

\begin{align} 
x^{(l+1)} = proxg(x^{(l)} - eta^{(l)} \nabla f, eta^{(l)})
\end{align}

where $\eta^{(l)}$ is the (variable) step-size at $l$th iteration; and proxg, 
$\nabla f$ denotes proximal operator of $g(x)$ and gradient operator of $f(x)$ 
respectively.

The iterations converges to a fixed point given that 

\begin{align}
$0 < \eta^{(l)} < 2/ \mathscr{L}(\nabla f)$, 
\end{align}

where $\mathscr{L}(\nabla f)$ is  Lipschitz constant of $\nabla f$\cite{combettes2005signal}. 
Since $\mathscr{L}(\nabla f)$ for the problem is not known, back-tracking line-search is
used to guarantee the stability of the algorithm. Along with that, in order to accelerate 
the the convergence, we initialized the step-sizes using "adaptive" BB-method\cite{zhou2006gradient}. 
The implementation is largely inspired by FASTA solver \cite{goldstein2014field}.
