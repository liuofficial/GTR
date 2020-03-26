function T = get_coef_lab(train_lab)
nClass = length(unique(train_lab));
train_size = length(train_lab);
T = zeros(nClass, train_size);
cls = unique(train_lab);
for k = 1 : nClass,
    c = cls(k);
    T(k, train_lab==c) = 1;
end
%T = sqrt(T*T') \ T;
end