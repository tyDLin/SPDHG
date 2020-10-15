# Stochastic Primal-Dual Hybrid Gradient for Compositely Regularized Minimization

These codes provide implementations of solvers for large-scale compositely regularized minimization using stochastic primal-dual hybrid gradient method. 

# About

In this paper, we propose a stochastic Primal-Dual Hybrid Gradient (PDHG) approach for solving a wide spectrum of regularized stochastic minimization problems, where the regularization term is composite with a linear function. It has been recognized that solving this kind of problem is challenging since the closed-form solution of the proximal mapping associated with the regularization term is not available due to the imposed linear composition, and the per-iteration cost of computing the full gradient of the expected objective function is extremely high when the number of input data samples is considerably large. 

Our new approach overcomes these issues by exploring the special structure of the regularization term and sampling a few data points at each iteration. Rather than analyzing the convergence in expectation, we provide the detailed iteration complexity analysis for the cases of both uniformly and non-uniformly averaged iterates with high probability. This strongly supports the
good practical performance of the proposed approach. Numerical experiments demonstrate that the efficiency of stochastic PDHG, which outperforms other competing algorithms, as expected by the high-probability convergence analysis.

# Codes

Implementations in MATLAB are provided, including graph-guided logistic regression and graph-guided regularized logistic regression.  

# References

On Stochastic Primal-Dual Hybrid Gradient Approach for Compositely Regularized Minimization. L. Qiao, T. Lin, Y-G. Jiang, F. Yang, W. Liu and X. Lu. ECAI'2016. 

On the Iteration Complexity Analysis of Stochastic Primal-Dual Hybrid Gradient Approach with High Probability. L. Qiao, T. Lin, Q. Qin and X. Lu. Neurocomputing, 307: 78-90, 2018.
