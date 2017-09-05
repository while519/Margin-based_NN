%% convert the train/test data split of the proposed algorithm input into RTM input
% Provided: training pairs indexes; testing pairs indexes; co-occurrence
% data matrix

% preprocessing
teIdxl_ = teIdxl + 1;
trIdxl_ = trIdxl + 1;
teIdxr_ = teIdxr + 1;
trIdxr_ = trIdxr + 1;

[D, W] = size(X);
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

save('cora_split.mat', 'Atrain', 'd', 'D', 'N', 'w', 'W', 'word', 'teIdxl_', 'teIdxr_', 'webpage_classnames', 'webpage_ids');