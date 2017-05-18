% solve: argmax log(det X) - tr(X . S) - \lambda \|X\|_1 : X \in S_+^(pxp)
% dual : argmin -log(det (S+U)) - p : \|U\|_infty \leq \lambda
% dual : argmin -log(det W) - p : \|W-S\|_infty \leq \lambda

% input: S = covariance matrix
%        opt.lambda = l1 regularization 
%        opt.diag = diagonal addition (default opt.lambda)
%        opt.maxiter = maximum number of iterations
%        opt.W0 = starting point

function [W,invW,adj] = graphical_lasso(S, lambda, diag_add, maxiter, W0)
    [p,~] = size(S);
    if (p ~= size(S,2)) 
        error('S must be square');
    end
    
    if nargin < 2
        lambda = 0.01;
    end
    
    if nargin < 3
        diag_add = lambda;
    end
    
    if nargin < 4
        maxiter = 10;
    end
    
    if nargin < 5
        W = S + diag_add*eye(p);
    else
        W = W0;
    end
    
    old_invW = inv(W);
    for iter=1:maxiter
        for j=1:p
            u = S(:, j); u(j) = [];
            V = W; V(:,j) = []; V(j, :) = [];
            beta = solve_column;
            y = V*beta;
            if norm(y-u,inf) > lambda+1e-10,
                display(sprintf('%d %d %f not feasible', iter, j, norm(y-u,inf)));
            end
            y = [y(1:j-1); W(j,j); y(j:end)];
            W(:,j) = y; W(j, :) = y';
        end
        invW = inv(W);

        if norm(old_invW(:)-invW(:),inf) < 1e-16,
            display(sprintf('iter=%d Change too small', iter))
            break;
        end
        old_invW = invW;
        
        duality_gap = trace(invW * S)-p+lambda*sum(sum(abs(invW)));
        display(sprintf('iter=%d gap=%f', iter, duality_gap));        
    end

    adj = zeros(size(invW));
    adj(W > 1e-6) = 1;
    adj(W < -1e-6) = -1;
    
    % lasso(V, u, lambda)
    function beta = solve_column
        solve_maxiter=1000;
        beta=zeros(p-1,1);
        for solve_iter=1:solve_maxiter
            old_beta = beta;
            for k=1:p-1
                idx = [1:k-1 k+1:p-1];
                x = u(k) - V(k,idx)*beta(idx);
                beta(k) = soft_threshold(x, lambda) / V(k,k);
            end
            if norm(old_beta-beta,2) < 1e-16, break; end
        end
        %solve_iter
    end
    
    function s = soft_threshold(x,rho)
        if abs(x) < rho
            s = 0;
        else
            s = sign(x) * (abs(x)-rho);
        end
    end
end
