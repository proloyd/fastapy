# fastapy
Python implementation of fast adaptive proximal gradient descent algorithm

Proximal gradient descent (also known as forward backward splitting or FBS) 
method is a way to solve high-dimensional optimization problems of form: 
\begin{align} \label{eq:PGD1}
\min_{x \in \mathbbm{R}^n} f(x) + g(x)
\end{align}
where $f(x)$ is convex and differentiable non-differentiable but $g(x)$ is typically not smooth but convex function. It is a two stage method which addresses two terms in \eqref{eq:PGD1} separately. Using this technique, updating $\boldsymbol{\Theta}$ can be done iteratively as described next. At $l$th iteration of the algorithm, we have:
\begin{align} \label{eq:PDG_iterations1}
&\widetilde{\boldsymbol{\Theta}} = \boldsymbol{\Theta}^{(l)} - \eta^{(l)} \mathbf{L}^{\top}\boldsymbol{\Sigma}_v^{-1}\left( \mathbf{L} \boldsymbol{\Theta}^{(l)}\widetilde{\mathbf{E}} - \mathbf{Y} \right) \widetilde{\mathbf{E}}^{\top} \\\label{eq:PDG_iterations2}
&\boldsymbol{\Theta}^{(l+1)} = \mathcal{S}_{\eta^{(l)}\mu}\left( \widetilde{\boldsymbol{\Theta}} \right)
\end{align}
where $\eta^{(l)}$ is the (variable) step-size at $l$th iteration and $\mathcal{S}_{\nu}:\mathbbm{R}^{N\times M} \rightarrow \mathbbm{R}^{N\times M}$ is `soft-thresholding' operator, whose $(m,n)$th component is given by:
\begin{align}
\left(\mathcal{S}_{\nu}\left( \widetilde{\boldsymbol{\Theta}} \right)\right)_{m,n} := \text{sgn}\left({\theta}^{(m)}_n\right)\max \left( \left\vert{\theta}^{(m)}_n\right\vert -\nu, 0 \right).
\end{align}
The iterations converges to a fixed point given that $0 < \eta^{(l)} < 2/ \mathscr{L}(\nabla f)$, where $\mathscr{L}(\nabla f)$ is  Lipschitz constant of $\nabla f$\cite{combettes2005signal}. Since $\mathscr{L}(\nabla f)$ for the problem is not known, we used back-tracking line-search to guarantee the stability of the algorithm, where after each iteration we checked the following condition:
\begin{align} \label{eq:stepsize_condition}
f(&\boldsymbol{\Theta}^{(l+1)}) < f(\boldsymbol{\Theta}^{(l)}) + \Big\langle \boldsymbol{\Theta}^{(l+1)} - \boldsymbol{\Theta}^{(l)}, \mathbf{L}^{\top}\boldsymbol{\Sigma}_v^{-1} \nonumber \\ 
&\times \left( \mathbf{L} \boldsymbol{\Theta}\widetilde{\mathbf{E}} - \mathbf{Y} \right) \widetilde{\mathbf{E}}^{\top}\Big\rangle + \frac{1}{2\eta^{(l)}}\Vert \boldsymbol{\Theta}^{(l+1)} - \boldsymbol{\Theta}^{(l)} \Vert_{F}^2.
\end{align} 
In case the condition is violated we decrease the step-size $\eta^{(l)}$ until \eqref{eq:stepsize_condition} satisfied. Along with that, in order to 
accelerate the the convergence, we initialized the step-sizes using ``adaptive" BB-method\cite{zhou2006gradient}. The implementation is largely inspired by FASTA solver \cite{goldstein2014field}, but especially tailored to solve \eqref{eq:PGD} fast and efficiently.
