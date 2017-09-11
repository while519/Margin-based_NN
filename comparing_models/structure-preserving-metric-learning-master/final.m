%% Obtaining the PR curve related testing scores
trIdxl_ = trIdxl + 1; 
trIdxr_ = trIdxr + 1;


Atest = zeros(D,D);
for ii = 1 : length(teIdxl_)
    Atest(teIdxl_(ii), teIdxr_(ii)) = 1;
end
Atest = Atest | Atest';
c = -Dist;

testSet = setdiff(1 : D, [trIdxl_(:) ; trIdxr_(:)]);
score = [];
target = [];
idxlSet = [];
idxrSet = [];

for idxl = 1 : D
    for idxr = 1 : D
        if idxl ~= idxr
            if ismember(idxl, testSet) | ismember(idxr, testSet)
                score = [score; c(idxl, idxr)];
                target = [target; Atest(idxl, idxr)];
                idxlSet = [idxlSet; idxl];
                idxrSet = [idxrSet; idxr];
            end
        end
    end
end

prec_rec(score, target);