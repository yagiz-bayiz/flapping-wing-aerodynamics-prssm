clear
clc
close all

input_dir = '../Data/Input/';
load([input_dir, 'flapping_wing_aerodynamics.mat']);                   % Load Lasso results and model

num_exp = 548;
train_ind = 1:512;
test_ind = 513:548;



%% Kinematic Features
AoA = ds_u_raw(:,:,1);
vw_abs = ds_u_raw(:,:,2);
aw_x = ds_u_raw(:,:,3);
aw_y = ds_u_raw(:,:,4);
aw_z = ds_u_raw(:,:,5);
velrot = ds_u_raw(:,:,6);
accrot = ds_u_raw(:,:,7);

[vw_x, vw_y] = pol2cart(AoA+pi/2, vw_abs);

ds_uLR_raw = [];
ds_uLR_raw = cat(3, ds_uLR_raw, ...
                vw_abs.^2,...
                AoA.*vw_abs.^2,...
                AoA.^2.*vw_abs.^2,...
                AoA.^3.*vw_abs.^2, ...
                velrot.*vw_abs,...
                aw_x,...
                aw_y,... 
                velrot.*vw_x,...
                velrot.*vw_y,...
                accrot,...
                velrot.*abs(velrot));

inputLR_train = reshape(permute(ds_uLR_raw(train_ind,:,:), [3, 2, 1]) , size(ds_uLR_raw,3), [])';
outputLR_train = reshape(permute(ds_y(train_ind,:,:), [3, 2, 1]) , size(ds_y,3), [])';

%% Lasso Fit
LRmodel = [];
for i = 1:size(outputLR_train,2)
    [B, FitInfo] = lasso(inputLR_train, outputLR_train(:,i),'CV',10,'Alpha',0.75,'PredictorNames',{'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11'});
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    LRmodel = [LRmodel, coef];
end

lr_y = [];
for i = 1:num_exp
    x = permute(ds_uLR_raw(i,:,:), [2, 3, 1]);
    y = x*LRmodel;
    lr_y = cat(3, lr_y, y);
end
lr_y = permute(lr_y, [3, 1, 2]);

%% Save Data
save([input_dir,'flapping_wing_aerodynamics_lasso_fit.mat'], 'ds_uLR_raw', 'ds_y', 'lr_y', 'LRmodel');