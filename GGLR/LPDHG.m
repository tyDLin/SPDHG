%Algorithm LPDHG_GGLR
function outputs = LPDHG(samples, labels, opts)
s = opts.s; F = opts.F; mu = opts.mu;  L = opts.L;
checki = opts.checki; epochs = opts.epochs;

% initialization
t = cputime; 
xs = []; times = []; iters = [];
[d,N]  = size(samples); x = zeros(d,1); y = zeros(d,1);xbar = zeros(d,1);
%code opt
nuFT = mu*F'; snuF = s*mu*F;

done = false; k = 0;
while ~done
    r= 1/(sqrt(k+1) + L);
    
    % LPDHG
    y = min( 1, max(-1,snuF*x+y) );
    
    sum_nabla_lx = 0;
    for idx=1:N
        temp_exp  = exp(-labels(idx)*(samples(:,idx)'*x));
        if isnan(temp_exp)
            temp_var = -labels(idx);
        else
            temp_var = -labels(idx)*(temp_exp/(1+temp_exp));
        end
    
        sum_nabla_lx = sum_nabla_lx + temp_var*samples(:,idx);
    end    
    x = x - r*( sum_nabla_lx/N + nuFT*y);  
    
    % uniformly averaging
    xbar   = (k*xbar + x)/(k+1);
    
    % log
    xs = [xs xbar];
    iters        = [iters k+1];
    time_solving = cputime - t;
    times        = [times time_solving];
    
    %terminate condition check
    k = k + 1;
    if k>epochs
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

