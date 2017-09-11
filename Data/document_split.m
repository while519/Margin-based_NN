%% Fully split the data into train/test partitions
%
[D,W] = size(X);
CV0 = cvpartition(D, 'KFold', 5);

trIdx = CV0.training(1);
teIdx = CV0.test(1);

    
tr_indexes = find(trIdx);
Lia = ismember(I, tr_indexes);
trI_indicator = logical(prod(Lia,2));
teI = I(~trI_indicator, :);
trI = I(trI_indicator, :);

trIdxl = trI(:, 1);
trIdxr = trI(:, 2);

teIdxl = teI(:, 1);
teIdxr = teI(:, 2);

%%
%
teIdxl_ = teIdxl + 1;
trIdxl_ = trIdxl + 1;
teIdxr_ = teIdxr + 1;
trIdxr_ = trIdxr + 1;

Atrain = zeros(D,D);    % Adjacency matrix for the training set 
for ii = 1 : length(trIdxl)
    Atrain(trIdxl_(ii), trIdxr_(ii)) = 1; 
end
Atrain = logical(sparse(Atrain)); 

d = [];
w = [];
x = find(X');
N = sum(X(:));
for nn = 1 : sum(X(:))
    d = [d ; floor(x(nn) / W) + 1  ];
    w_indice = rem(x(nn), W);
    if ~w_indice
        w_indice = W;
    end
    w = [w; w_indice];
end

save('webkb_split_v01.mat', 'Atrain', 'd', 'D', 'N', 'w', 'W', 'word', 'teIdxl_', 'teIdxr_', ...
        'trIdxl_', 'trIdxr_', 'trIdxl', 'trIdxr', 'teIdxl', 'teIdxr', 'webpage_classnames', 'webpage_ids');