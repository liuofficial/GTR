function demo_gtr
% demo for the first kind of features, i.e., SD features
addpath('tensor_toolbox_2.6');
n = 1;
[img, img_gt, nClass, rows, cols, bands] = load_datas(n);
nts = 6; its = 10;
switch n
    case 1, type = 2; trr_lam = 1e-3;
        WH = [9 9];
end

ten_ind = compute_tensor_index(rows, cols, WH);

nt = 3; it = 2;
        [train_idx, test_idx] = load_train_test(n, type, nt, it);
        [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt, 0);
        
        t_begin = tic;
        train_win = ten_ind(:, Train.idx);
        train_win = reshape(train_win, prod(WH)*Train.size, 1);
        tm = img(:,train_win);
        A = reshape(tm, [bands, WH, Train.size]);
        T = get_coef_lab(Train.lab);
        indT = 1 : numel(WH)+2;
        indT(2) = [];
        T = permute(T, [indT 2]);

        opt = [];
        opt.lam = trr_lam; opt.verb = 1; opt.tol = 1e-3; opt.maxit = 200;
        opt.neg = 1; % better for kernel-based features
        [M] = trr_train(T, A, opt);
        P = TRR_test_bloc(M, img, ten_ind, WH, bands, size(T,1));
        [~,trr_pred] = max(P);
        trr_acc = class_eval(trr_pred(Test.idx), Test.lab);
        trr_time = toc(t_begin);
        disp(trr_acc);
end

function ten_ind = compute_tensor_index(rows, cols, WH)
if numel(WH) == 1
    W = WH^0.5; H = W;
else
    W = WH(1); H = WH(2);
end
im = 1 : rows*cols;
im = reshape(im, rows, cols);
im = padarray(im, [(W-1)/2+1 (H-1)/2+1], 'both', 'symmetric');
im([(W-1)/2+1 end-(W-1)/2],:) = [];
im(:,[(H-1)/2+1 end-(H-1)/2]) = [];
ten_ind = im2col(im, [W H]);
end

function P = TRR_test_bloc(M, im, ten_ind, WH, fea_size, nClass)
% X: L x (W x H) x S x N
test_win = ten_ind(:, :);
Testsize = size(test_win,2);
bloc = 2500;
nbloc = ceil(Testsize/bloc);
P = zeros(nClass, Testsize);
for i = 1 : nbloc
    idx = (i-1)*bloc+1 : min(i*bloc, Testsize);
    testwin = reshape(test_win(:,idx), prod(WH)*length(idx), 1);
    tm = im(:,testwin);
    X = reshape(tm, [fea_size, WH, length(idx)]);
    P(:,idx) = trr_test(X, M);
end
end
