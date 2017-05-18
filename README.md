# StochasticPDHG
Stochastic Primal-Dual Hybrid Gradient for Compositely Regularized Minimization

These codes provide implementations of solvers for large-scale compositely regularized minimization using stochastic primal-dual hybrid gradient method. 

About

We consider a wide spectrum of regularized stochastic minimization problems, where the regularization term is composite
with a linear function. Examples of this formulation include graphguided regularized minimization, generalized Lasso and a class of l1-regularized problems. The computational challenge is that the closed-form solution of the proximal mapping associated with the regularization term is not available due to the imposed linear composition. Fortunately, the structure of the regularization term allows us to reformulate it as a new convex-concave saddle point problem which can be solved using the Primal-Dual Hybrid Gradient (PDHG) approach. However, this approach may be inefficient in realistic applications
as computing the full gradient of the expected objective function could be very expensive when the number of input data
samples is considerably large. To address this issue, we propose a Stochastic PDHG (SPDHG) algorithm with either uniformly or nonuniformly averaged iterates. Numerical experiments on different genres of datasets demonstrate that our proposed algorithm
outperforms other competing algorithms.

Codes

Implementations in MATLAB are provided, including graph-guided logistic regression and graph-guided regularized logistic regression.  

References

L. Qiao, T. Lin, Y-G. Jiang, F. Yang, W. Liu, X. Lu. On Stochastic Primal-Dual Hybrid Gradient Approach for Compositely Regularized Minimization. Proc. of the 22th ECAI Conference (2016).

L. Qiao, T. Lin, Y-G. Jiang, F. Yang, X. Lu, On Stochastic Primal-Dual Hybrid Gradient Approach for Compositely Regularized Minimization. Pattern Recognition Letter. Under Review. 
