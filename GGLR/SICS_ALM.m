function out = SICS_ALM(S,rho,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solve min  -log det X + < S, X> + rho*||X||_1
% i.e., min  -log det X + < S, X> + rho*||Y||_1
%       s.t.  X = Y.
% using the Alternating Linearization Method to solve
% the smooth problem (smoothed L1 norm) via the following iterations:
% (The Gauss-Seidel 2-splitting alg. in Goldfarb and Ma, 2009)
%
% (1) X^{k+1} := argmin_X f(X) +
%              g_sigma(Y^k)+<grad(g_sigma(Y^k)),X-Y^k>+||X-Y^k||_F^2/(2 mu)
%
% (2) Y^{k+1} := argmin_Y f(X^{k+1}) + <grad(f(X^{k+1})), Y-X^{k+1}>
%                         + ||Y -X^{k+1}||_F^2/(2 mu) + g_sigma(Y)
%
% Author: Shiqian Ma
% Date:   Dec. 17, 2009
% Last modified: Apr. 26, 2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(S,1);
X = eye(n,n); Y = zeros(n,n); gradgY = zeros(n,n);
mu = opts.mu0;
fc = inf; dualgap = inf;
sigma = opts.sigma;

for itr = 1: opts.mxitr

    %% update X
    Xp = X;
    W = Y/mu - gradgY - S; W = (W + W')/2;
%     [V, D] = mexeig(W); % change to mexeig for a faster eig-decomposiiton
    [V,D] = eig(W); 
    d = diag(D);
    gamma = (mu*d+sqrt((mu*d).^2+4*mu))/2;
    X = V*diag(gamma)*V';
    gradfX = -V*diag(1./gamma)*V' + S;

    %% update Y
    Yp = Y;
    Y = X-mu*gradfX-mu*min(rho,max(-rho,(X-mu*gradfX)/(sigma+mu)));
    gradgY = min(rho,max(-rho,Y/sigma));

    %% compute and print stats

    dualgap_plus_n_X = sum(sum(S.*X)) + rho* sum(sum(abs(X)));
    dualgap_X = dualgap_plus_n_X - n;

    dualgap_plus_n_Y = sum(sum(S.*Y)) + rho* sum(sum(abs(Y)));
    dualgap_Y = dualgap_plus_n_Y - n;

    %     dualgap = max(abs([dualgap_X, dualgap_Y]));

    nrmXmY = norm(X-Y,'fro');
    fp = fc;
    fc = -(sum(log(gamma)) - dualgap_plus_n_X);
    pinf = nrmXmY/max(norm(X,'fro'),norm(Y,'fro'));
    frel = abs(fp - fc)/max(abs([fp,fc,1]));
    Xrel = norm(X-Xp,'fro')/max([1,norm(X,'fro'),norm(Xp,'fro')]);
    Yrel = norm(Y-Yp,'fro')/max([1,norm(Y,'fro'),norm(Yp,'fro')]);

    if opts.record > 0
        fprintf('iter: %d, mu: %3.2e, pinf: %3.2e, frel: %3.2e, Xrel: %3.2e, Yrel: %3.2e, gap: %3.2e\n', itr, mu, pinf, frel, Xrel, Yrel, dualgap);
    end

    %% check is stop
    stop = ( frel < opts.tol_frel ) && ( Xrel < opts.tol_Xrel ) && ( Yrel<opts.tol_Yrel );
    % in every numDG steps, check duality gap
    if mod(itr,opts.numDG) == 0
        Lambda = gradgY;
%         [V,D] = mexeig(S+Lambda); d = diag(D);
        [V,D] = eig(S+Lambda); d = diag(D); 
        dualgap = -sum(log(gamma))+sum(sum(S.*X))+rho*sum(sum(abs(X))) - sum(log(d))-n;
        stop = stop || (dualgap < opts.tol_gap);
        mu = max(mu * opts.rmu, opts.muf);
    end

    if stop
        out.X = X; out.Y = Y; out.iter = itr; out.pinf = pinf;
        out.obj = sum(log(gamma)) - dualgap_plus_n_X;
        out.gapX = dualgap_X; out.gapY = dualgap_Y;
        out.gap = dualgap;
        return;
    end
end

out.X = X; out.Y = Y; out.iter = itr; out.pinf = pinf;
out.obj = sum(log(gamma)) - dualgap_plus_n_X;
out.gapX = dualgap_X; out.gapY = dualgap_Y;
out.gap = dualgap;
