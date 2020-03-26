function img = DPTailor(dat, rdown, rup)
% 2013-07-04
img = dat;
[lmax lmin] = linear(dat, rdown, rup);
img(dat<lmin) = lmin; img(dat>lmax) = lmax;
img = (img-lmin) / (lmax-lmin);
end

function [lmax lmin] = linear(X, a, b)
X = X(:);
x = sort(X);
L = length(x);
lmin = x(max(ceil(L*a),1));
lmax = x(min(floor(L*b),end));
end