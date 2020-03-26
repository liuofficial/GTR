function [Train, Test, Back, Ground] = set_train_test(train_idx, test_idx, img, img_gt, dat)
% 2013-12-24
% func: 0; data: 0
% dat = 1, include Train.dat...
if nargin < 5, dat = 1; end
img_size = length(img_gt); img_idx = 1 : img_size; 

Train.idx = train_idx; Test.idx = test_idx;
if dat == 1,
    Train.dat = img(:, train_idx); Test.dat = img(:, test_idx);
end
Train.lab = img_gt(train_idx)'; Test.lab = img_gt(test_idx)';
train_size = length(train_idx); test_size = length(test_idx);

back_idx = true(img_size,1); back_idx(train_idx) = false; back_idx(test_idx) = false;
back_idx = img_idx(back_idx);
Back.idx = back_idx;
if dat == 1,
    Back.dat = img(:, back_idx);
end
Back.lab = 0 .* back_idx;
back_size = length(back_idx);

ground_idx = true(img_size,1); ground_idx(train_idx) = false;
ground_idx = img_idx(ground_idx);
Ground.idx = ground_idx;
if dat == 1,
    Ground.dat = img(:, ground_idx);
end
Ground.lab = 0 .* ground_idx;
ground_size = length(ground_idx);

Train.size = train_size; Test.size = test_size;
Back.size = back_size; Ground.size = ground_size;
end