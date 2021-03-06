% Adaptive Stochastic ADMM proposed by Peilin Zhao
% Algorithm: Stochastic Ada_SADMM_diag
function outputs = Ada_SADMMdiag(samples, labels, opts)
F = opts.F; mu = opts.mu; beta = opts.beta;
max_it = opts.max_it; checki = opts.checki; a = opts.a;

% initialization
t = cputime; time_solving = 0;
xs = []; times = []; iters = []; Num_i  = 0;
[d,N] = size(samples);  rnd_pm = randperm(N);
x = zeros(d,1); y = zeros(d,1); xbar = zeros(d,1);
if d<5000
    I=eye(d); G = zeros(d);
else
    I=speye(d); G = sparse(d,d);
end
theta = y;
%code opt
betaFTF = beta*(F'*F); d_bound  = 1; 
eta = d_bound / sqrt(max_it); 

if isfield(opts,'scaling')
    scaling = opts.scaling;
else
    scaling = 1;
end
eta = eta*scaling;

done = false; k = 1;
while ~done
    % randomly choose a sample
    idx = rnd_pm(mod(k,N)+1);
    sample = samples(:,idx);
    label  = labels(idx);
    
    % Stochastic ADMM
    t_exp  = exp(-label*sample'*x);
    if isnan(t_exp)
        g = -label*sample;
    else
        g = -label*sample*(t_exp/(1+t_exp));
    end    
    
    if d<500
        G = G+diag(g.*g);
    else
        t_G = speye(d);
        t_G(1:(d+1):d*d) = g.*g;
        G = G + t_G;
    end
    
    S = sqrt(G);
    H = a*I + S;
    x = (betaFTF+H/eta)\(F'*(beta*y + theta) + H*x/eta - g); 
    y_hat = F*x - theta/beta;
    y = sign(y_hat).*max(0, abs(y_hat)-mu/beta);
    theta = theta- beta*(F*x-y);
    
    % log
    Num_i  = Num_i+1;
    if (Num_i >= checki)
        xs = [xs x];
        iters = [iters k];
        time_solving = cputime - t;
        times = [times time_solving];
        Num_i    = 0;
    end    
    % terminate condition check
    k = k + 1;
    if k>max_it
        done = true;
    end
end

trace = []; 
trace.checki = checki; 
trace.xs     = xs;
trace.times  = times;
trace.iters  = iters;

outputs.x      = x;
outputs.iter   = k; 
outputs.trace  = trace;
end

