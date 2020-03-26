function [trainidx testidx] = load_train_test(n, type, i, j)
% 2014-06-14
switch n,
    case 1, 
        location = 'indian_';
end

switch type,
    case 1,
        types = 'n';
    case 2,
        types = 'r';
end

load([location types num2str(i) '_' num2str(j) '.mat']);
trainidx = train_idx; testidx = test_idx;
end