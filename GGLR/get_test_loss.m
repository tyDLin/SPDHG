function test_loss = get_test_loss(s_test,l_test,x)
N = length(l_test);
logist = zeros(N,1);
Z = sparse(1:N,1:N,l_test,N,N)*s_test';
Zx = -Z*x; posind = (Zx > 0);
logist(posind) = 1 + exp(-Zx(posind));
logist(~posind) = 1 + exp(Zx(~posind));
test_loss = (sum(log(logist(~posind))) + sum(Zx(posind) + log(logist(posind))))/N;
end