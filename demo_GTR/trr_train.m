function [M] = trr_train(T, A, opt)
% 2019-01-05
% tensor ridge regression (TRR) for multivariate classification
% T: C x w x h x... J; A: L x W x H x... J; M{i}: i=1, 2, 3...
if nargin < 3, opt = []; end
% parameters for cross-validation
if ~isfield(opt,'cv'), opt.cv = 0; end
if ~isfield(opt,'lams'), opt.lams = 10.^(-6:2); end
if ~isfield(opt,'cvtrain'), opt.cvtrain = 0; end
% parameter setting for cross-validation and TRR
if ~isfield(opt,'lam'), opt.lam = 1e-3; end
if ~isfield(opt,'maxit'), opt.maxit = 200; end
if ~isfield(opt,'tol'), opt.tol = 1e-3; end
if ~isfield(opt, 'verb'), opt.verb = 1; end
%  parameter for Rank-R TRR
if ~isfield(opt, 'nR'), opt.nR = 1; end
% parameter for class-wise TRR
if ~isfield(opt, 'nC'), opt.nC = 1; end
% T and A should be tensors
T = tensor(T); A = tensor(A);
T = augT(T, A); % size of T is C x 1 x 1 x... J
% cross-validation
if opt.cv > 0 && opt.nR == 1 && opt.nC == 1
    M = cvFun(T, A, opt);
    if opt.cvtrain > 0 % train and then perform TRR
        if opt.verb > 0
            disp(M); disp('begin train...');
        end
        opt.lam = M.lam;
        M = mainFun(T, A, opt);
    end
else % perform TRR
    if opt.nR == 1 && opt.nC == 1, M = mainFun(T, A, opt); end
end
% cross-validation is not provided for Rank-R
nClass = T.size(1);
if opt.nR > 1 || opt.nC == nClass && opt.cv == 0, M = mainFunRC(T, A, opt); end
end
% ==============================================
function M = mainFunRC(T, A, opt)
% divide-and-conquer strategy
T = squeeze(T);
nClass = opt.nC;
Ms = cell(ndims(A)-1, opt.nR, opt.nC);
for c = 1 : nClass
    Tc = tensor(T.data(c,:));
    Tc = augT(Tc, A);
    if opt.nC==1, Tc = T; end
    Ts0 = mean(abs(Tc.data(:)));
    for r = 1 : opt.nR
        M = mainFun(Tc, A, opt);
        Ms(:,r,c) = M;
        Tc = Tc - squeeze(ttm(A, M, [1:length(M)]));
        Ts1 = mean(abs(Tc.data(:)));
        opt.lam = max(1e-6, opt.lam * ( Ts1 / Ts0 )^2);
        if opt.verb == 3, disp(norm(Tc)); end
    end
end
M = Ms;
end
% ------------------ main function for TRR ---------------------------
function M = mainFun(T, A, opt)
if nargin < 3, opt = []; end
% parameter setting
if ~isfield(opt,'neg'), neg = 0; else neg = opt.neg; end
if ~isfield(opt,'lam'), lam = 1e-3; else lam = opt.lam; end
if ~isfield(opt,'maxit'), maxit = 200; else maxit = opt.maxit; end
if ~isfield(opt,'tol'), tol = 1e-3; else tol = opt.tol; end
if ~isfield(opt, 'verb'), verb = 1; else verb = opt.verb; end
% T and A should be tensors
T = tensor(T); A = tensor(A);
Andim = ndims(A);
T = augT(T, A); % size of T is C x 1 x 1 x... J
neg = repmat(neg, 1, Andim-1); % convenient usage
neg(1) = 0; % the nonnegative constraint is not imposed for the first dimension
% init output M
M = cell(Andim-1,1);
for k = 1 : Andim-1 % init
    M{k} = initM(A, T, A.size, T.size, lam, A.size(end), k);
end
% for k = 2 : Andim-1 % init
%     %M{k} = ones(1, A.size(k)) ./ sqrt(A.size(k));
%     %M{k} = rand(1, A.size(k));
%     M{k} = norm2one(M{k});
%     disp(norm(M{k},'fro')^2);
% end
% ********** main **************-
tols = ones(Andim-1,1);
for iter = 1 : maxit
    M0 = M;
    for k = 1 : Andim-1
        valMM = prodM2(M, k);
        M{k} = updateM(M, A, T, lam*valMM, neg(k), k);
    end
    if rem(iter,10)==0, if verb == 1, fprintf('.'); end; end
    for k = 1 : Andim-1
        tols(k) = norm(M0{k}-M{k},'fro') / norm(M0{k},'fro');
        %fprintf('%f ',norm(M{k},'fro')^2);
    end
    %disp(' ');
    if verb == 3, disp(norm(T - ttm(A, M, [1:length(M)]))); end
    if verb == 2, disp(tols'); end
    if iter > 5, if prod(tols < tol), break; end; end
end
if verb > 0, disp(['TRR total no.:', num2str(iter)]); end
end
% =========================================
% init M by RR
function val = initM(A, T, ASize, TSize, lam, J, k)
n_mode = ndims(A);
ind = 1 : n_mode-1;
ind(k) = [];
A_ = A.data; T_ = T.data;
for i = 1 : length(ind)
    A_ = mean(A_, ind(i));
    T_ = mean(T_, ind(i));
end
A_ = reshape(A_, ASize(k), J);
T_ = reshape(T_, TSize(k), J);
val = T_*A_' / (A_*A_' + lam*eye(ASize(k)));
end
% ||M1||_F^2 + ||M2||_F^2 + ||M3||_F^2 + ...
function val = updateM(M, A, T, lam, neg, k)
nDim = ndims(A);
Mk = M;
Mk(k) = [];
ind = 1 : nDim-1;
ind(k) = [];
Bk = ttm(A, Mk, ind);
Bk = tenmat(Bk,k); Tk = tenmat(T,k);
Bk = Bk.data; Tk = Tk.data;
BkBkt = Bk * Bk' + lam * eye(size(Bk,1));
TkBkt = Tk * Bk';
val = TkBkt / BkBkt;
if neg == 1, val = max(val,0); end
% if neg == 1
%     val = fnnls(BkBkt, TkBkt')';
% else
%     val = TkBkt / BkBkt;
% end
end
% compute ||M||_F^2
function val = prodM2(M, k)
MSize = length(M);
ind = 1 : MSize;
ind(k) = [];
val = norm(M{ind(end)}, 'fro')^2;
for i = length(ind)-1 : -1 : 1
    val = val * norm(M{ind(i)}, 'fro')^2;
end
end
% =======================================
% main function for cross-validation
function val = cvFun(T, A, opt)
lamLen = length(opt.lam);
verb = opt.verb; lams = opt.lams;
switch lamLen
    case 2, [x,y] = meshgrid(lams(:),lams(:)); lams = [x(:) y(:)];
    case 3
end
lamSize = length(opt.lams) ^ lamLen;
answ = zeros(lamSize,lamLen+1);
if verb == 1, fprintf('begin %d cv: %4d', lamSize, 0); end
for i = 1 : lamSize
    if verb == 1, fprintf('\b\b\b\b%4d', i); end
    OA = cvtrain(T, A, lams(i,:), opt);
    answ(i,:) = [lams(i,:) OA];
    if verb == 3, disp([num2str(lams(i)) ', OA: ', num2str(OA)]); end
end
if verb == 1, fprintf('\n'); end
[~,i] = max(answ(:,1+lamLen));
val.lam = answ(i,1:end-1); val.OA = answ(i,end); val.answ = answ;
end
% cell function for cross-validation
function OA = cvtrain(T, A, lam, opt)
kfold = opt.cv; verb = opt.verb;
nmode = ndims(A);
trainSize = A.size(end);
randTrain = randperm(trainSize);
T = permute_tensor_indice(T, nmode, randTrain);
A = permute_tensor_indice(A, nmode, randTrain);
cvBloc = ceil(trainSize / kfold);
OA = 0;
if verb > 1, fprintf('begin cv: %4d',0); end
for i = 1 : kfold
    if verb > 1, fprintf('\b\b\b\b%4d', i); end
    a = (i-1)*cvBloc+1; b = min(trainSize, i*cvBloc);
    if b<a, break; end
    iTestIdx = a : b;
    iTrainIdx = 1 : trainSize; iTrainIdx(iTestIdx) = [];
    iT = permute_tensor_indice(T, nmode, iTrainIdx);
    iA = permute_tensor_indice(A, nmode, iTrainIdx);
    iLab = permute_tensor_indice(T, nmode, iTestIdx);
    iX = permute_tensor_indice(A, nmode, iTestIdx);
    opt.lam = lam; opt.verb = 0;
    iM = mainFun(iT, iA, opt);
    iP = trr_test(iX, iM);
    [~,pred] = max(iP);
    [~,test_lab] = max(squeeze(iLab.data));
    OA = OA + sum(pred == test_lab);
end
OA = OA / trainSize * 100;
if verb > 1, fprintf('\n'); end
end
% ===========================================
function P = trr_test(X, M)
n_mode = ndims(X);
X = tensor(X);
P = ttm(X, M, [1:length(M)]);
P = P.data;
str_ind = 'P(:,';
for i = 1 : n_mode-2
    str_ind = [str_ind '1,'];
end
str_ind = [str_ind ':)'];
P = eval(str_ind);
P = reshape(P,size(P,1),size(P,n_mode));
end
function val = permute_tensor_indice(X, ind, inds)
% X(:,...,inds,...,:)
modes = ndims(X);
X = X.data;
strInd = 'X(';
for i = 1 : modes
    if i == ind
        strInd = [strInd 'inds'];
    else
        strInd = [strInd ':'];
    end
    if i < modes, strInd = [strInd ',']; end
end
strInd = [strInd ')'];
val = eval(strInd);
val = tensor(val);
end
function T = augT(T, A)
Tndim = ndims(T); Andim = ndims(A);
if Tndim < Andim % size of T is C x 1 x 1 x... J
    T = squeeze(T.data);
    ind = 1 : Andim;
    ind(2) = [];
    T = permute(T, [ind 2]);
    T = tensor(T);
end
end
function val = norm2one(x)
val = x ./ sqrt(sum(x(:).^2));
end