function outputs = Fast_SADMM(samples, labels, opts)
max_it = opts.max_it; checki = opts.checki; nu = opts.mu;
gamma = opts.gamma; beta = 1; para2 = 0.01;

if isfield(opts,'scaling')
    scaling = opts.scaling;
else
    scaling = 1;
end
if size(opts.F,1) <5000
    F = [opts.F;eye(size(opts.F,1))];
else
    F = [opts.F;speye(size(opts.F,1))];
end

% initialization
rng('default')
rnd_pm = randperm(length(labels));
Xtr = samples(:,rnd_pm);
ytr = labels(rnd_pm);
[d,N] = size(Xtr);
x = zeros(d,1); y = zeros(size(F,1),1); zeta = zeros(size(F,1),1);
g = zeros(d,1); hist = zeros(N,1);  % historical value
sumx = zeros(d,1);  % summation of all x
F = sparse(F); allx = zeros(d,N); bFt = beta*F';

xs = []; thetimes=[];iters = []; Num_i  = 0;
time_temp = cputime; time_solving = 0;

done = false; k = 1;
while ~done
    % randomly choose a sample
    idx = rnd_pm(mod(k,N)+1);
    sig = -ytr(idx)/(1+exp(ytr(idx)*(x'*Xtr(:,idx))));
    
    g = g + Xtr(:,idx)*(sig-hist(idx));
    hist(idx) = sig;
    sumx = sumx + x -allx(:,idx);
    allx(:,idx) = x;
    eta = scaling*para2/sqrt(k);
    x =   x-eta*(sig*Xtr(:,idx)+gamma*x-bFt*(zeta-F*x + y));
    
    FxMz = F*x-zeta;
    y = sign(FxMz).*max(abs(FxMz)-nu/beta,0);
    zeta = y-FxMz ;
    
    % log
    Num_i  = Num_i+1;
    if (Num_i >= checki)
        xs       = [xs x];
        iters    = [iters k];
        time_solving = cputime - time_temp;
        thetimes = [thetimes time_solving];
        Num_i    = 0;
    end
    
    % terminate condition check
    k = k + 1;
    if k > max_it
        done = true;
    end
end

% return value
trace        = [];
trace.xs     = xs;
trace.times  = thetimes;
trace.iters  = iters;

outputs.x      = x;
outputs.iter   = k;
outputs.trace  = trace;
end
