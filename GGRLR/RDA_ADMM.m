% RDA-ADMM proposed by Suzuki, Taiji in http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2013_suzuki13.pdf
function outputs = RDA_ADMM(samples, labels, opts)
F = opts.F; mu = opts.mu; beta = opts.beta; betaF = beta*F; gamma = opts.gamma;
max_it = opts.max_it; checki = opts.checki; eta = opts.eta; 
if isfield(opts,'scaling')
    scaling = opts.scaling;
else
    scaling = 1;
end
eta = eta*scaling;

% initialization
t = cputime;
xs = []; times = []; iters = []; Num_i  = 0; 
[d,N] = size(samples);  rnd_pm = randperm(N);
x = zeros(d,1); lambda = zeros(d,1);  
xbar = zeros(d,1); ybar = zeros(d,1); lambdabar = zeros(d,1); gbar = zeros(d,1);

done = false; k = 0;
while ~done
    % randomly choose a sample
    idx = rnd_pm(mod(k,N)+1);
    sample = samples(:,idx);
    label  = labels(idx);
    
    % RDA-ADMM
    t_exp  = exp(-label*sample'*x);
    if isnan(t_exp) || isinf(t_exp)
        g = -label*sample;
    else
        g = -label*sample*(t_exp/(1+t_exp));
    end
    gbar = (k*gbar + g + gamma*x)/(k+1);
    
    x = -(k+1)*(gbar - F'*(lambdabar - betaF*xbar + beta*ybar))/eta;
    Fx = F*x;
    y = wthresh(Fx - lambda/beta,'s',mu/beta);
    lambda = lambda - beta*(Fx - y);
    xbar = (k*xbar + x)/(k+1);
    ybar = (k*ybar + y)/(k+1);
    lambdabar = (k*lambdabar + lambda)/(k+1);
    
    % log
    Num_i  = Num_i+1;
    if (Num_i >= checki)
        xs = [xs xbar];
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

outputs.x      = xbar; 
outputs.iter   = k; 
outputs.trace  = trace;
end

