%% Calculate mean average rank for the testing pairs
%

trIdxl_ = trIdxl + 1;
trIdxr_ = trIdxr + 1;


c = c - diag(diag(c));
N_test = length(teIdxl_);
RANK = [];

for ii = 1 : N_test
    ic = tiedrank(-c(teIdxl_(ii), :));
    RANK = [RANK; ic(teIdxr_(ii))];
    ic = tiedrank(-c(teIdxr_(ii), :));
    RANK = [RANK; ic(teIdxl_(ii))];
end
MAP = mean(RANK)


N_train = length(trIdxl_);
RANK = [];

for ii = 1 : N_train
    ic = tiedrank(-c(trIdxl_(ii), :));
    RANK = [RANK; ic(trIdxr_(ii))];
    ic = tiedrank(-c(trIdxr_(ii), :));
    RANK = [RANK; ic(trIdxl_(ii))];
end
MAP = mean(RANK)