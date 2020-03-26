function opt = class_eval(dat_lab, lab)
% 2013-12-24
% func: 0; data: 0
cls = unique(lab);
nClass = length(cls);
opt.ratio = zeros(1,nClass); opt.OA = 0; opt.num = zeros(nClass);
l = 1;
for k = cls(:)'
    idx = (lab == k);
    k_cls = sum(idx);
    k_crt = sum(dat_lab(idx) == k);
    opt.ratio(l) = k_crt / k_cls * 100;
    opt.OA = opt.OA + k_crt;
    j = 1;
    for i = cls(:)'
        opt.num(j,l) = sum(dat_lab(idx)==i);
        j = j + 1;
    end
    l = l + 1;
end
opt.OA = opt.OA / length(lab) * 100;

n = sum(opt.num(:));                     % Total number of samples
PA     = sum(diag(opt.num));

% Estimated Overall Cohen's Kappa (suboptimal implementation)
npj = sum(opt.num,1);
nip = sum(opt.num,2);
PE  = npj*nip;
if (n*PA-PE) == 0 && (n^2-PE) == 0
    % Solve indetermination
    opt.KA = 1;
else
    opt.KA  = (n*PA-PE)/(n^2-PE);
end

opt.AA = mean(opt.ratio);

% quantity disagreement and allocation disagreement
opt.Q = 0.5 * sum(abs(npj(:) - nip(:))) / n;
opt.A = sum(min([npj(:) - diag(opt.num); nip(:) - diag(opt.num)],2)) / n;
end