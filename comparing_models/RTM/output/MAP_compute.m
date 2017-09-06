%% Calculate mean average rank for the testing pairs
%


c = c - diag(diag(c));
N_test = length(teIdxl_);
RANK = [];

for ii = 1 : N_test
    ic = tiedrank(-c(teIdxl_(ii), :));
    RANK = [RANK; ic(teIdxr_(ii))];
end
MAP = mean(RANK)