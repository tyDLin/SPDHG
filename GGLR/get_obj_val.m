function obj_val = get_obj_val(s_train,l_train,x, F, mu)
N = length(l_train);
logist = zeros(N,1);
Z = sparse(1:N,1:N,l_train,N,N)*s_train';
Zx = -Z*x; posind = (Zx > 0);
logist(posind) = 1 + exp(-Zx(posind));
logist(~posind) = 1 + exp(Zx(~posind));
obj_val = (sum(log(logist(~posind))) + sum(Zx(posind) + log(logist(posind))))/N;
%obj_val = sum(log(1+exp(-l_train.*(s_train'*x))))/length(l_train);

obj_val = obj_val + mu*sum(sum(abs(F*x)));

end