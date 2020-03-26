function [img img_gt nClass rows cols bands] = load_datas(n)
% 2014-06-14
switch n
    case 1,
        load indian_info.mat;
        img = indian_im; img_gt = indian_gt;
        img([104:108 150:163 220],:) = [];
        nClass = 16;
        rdown = 0.001; rup = 0.999;
end

[rows, cols] = size(img_gt); bands = size(img,1);
img_gt = img_gt(:);
img = DPTailor(img, rdown, rup);
end