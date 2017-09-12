%% configuring and evaluating the SPML model
%
clear
clc
rng(213)
load data/webkb_split_v01.mat

%% normalize data

X = bsxfun(@rdivide, X, sum(X,2));
X(isnan(X)) = 0;

%% symmetrize links

A = Atrain | Atrain';

%% run spml

params = [];
params.lambda = 1e-6;
params.maxIter = 5000;
params.printEvery = 100;
params.project = 'final';
params.diagonal = true;
% turn off 'diagonal' full matrix (richer metric)
%params.diagonal = false;


model = spml(X', A, params);

%% Compute mean average rank
Dist = metricDistanceMask(X', model.M, ones(D,D));
N_test = length(teIdxl_);
RANK = [];

for ii = 1 : N_test
    I = double(teIdxl_(ii)) * ones(D, 1);
    J = 1 : D;
    %Dist = metricDistanceMask(X', model.M, sparse(I,J,true,D,D));  
    ic = tiedrank(Dist(teIdxl_(ii), :)) -1;
    RANK = [RANK; ic(teIdxr_(ii))];
end
MAP = mean(RANK)


trIdxl_ = trIdxl + 1;
trIdxr_ = trIdxr + 1;
N_train = length(trIdxl_);
RANK = [];

for ii = 1 : N_train
    I = double(trIdxl_(ii)) * ones(D, 1);
    J = 1 : D;
    %Dist = metricDistanceMask(X', model.M, sparse(I,J,true,D,D));  
    ic = tiedrank(Dist(trIdxl_(ii), :)) -1;
    RANK = [RANK; ic(trIdxr_(ii))];
end
MAP = mean(RANK)