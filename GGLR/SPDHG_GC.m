% uniformly general convex SPDHG
function outputs  = SPDHG_GC(samples, labels, opts)
mu = opts.mu; L = opts.L; s  = opts.s; F = opts.F; 
max_it = opts.max_it; checki = opts.checki; eta = 1;
if isfield(opts,'scaling')
    scaling = opts.scaling;
else
    scaling = 1;
end
eta = scaling*eta;

% initialization
time = cputime; 
xs = []; times = []; iters = []; Num_i  = 0; 
[d,N]  = size(samples); rnd_pm = randperm(N);
x = zeros(d,1); y = zeros(d,1); xbar = zeros(d,1);

%code opt
nuFT = mu*F'; snuF = s*mu*F;

done = false; k = 0;
while ~done
    r  = eta/(sqrt(k+1) + L);    
    % Stochastic PDHG
    y = min(1, max(-1,snuF*x+y));
    
    % randomly choose a sample
    idx    = rnd_pm(mod(k,N)+1);
    sample = samples(:,idx);
    label  = labels(idx);
    t_exp  = exp(-label*sample'*x);
    if isnan(t_exp) || isinf(t_exp)
        g = -label*sample;
    else
        g = -label*sample*(t_exp/(1+t_exp));
    end
    x      = x - r*(g + nuFT*y);
    
    % uniformly averaged
    xbar   = (k*xbar + x)/(k+1);
    
    % log
    Num_i  = Num_i+1;
    if (Num_i >= checki)
        xs = [xs xbar];
        iters = [iters k];
        time_solving = cputime - time;
        times = [times time_solving];
        Num_i    = 0;
    end
    
    %terminate condition check
    k = k + 1;
    if k>max_it
        done = true;
    end
end

% return value
trace = []; 
trace.checki = checki; 
trace.xs     = xs; 
trace.times  = times;
trace.iters  = iters;

outputs.x      = xbar;
outputs.iter   = k; 
outputs.trace  = trace;
end

