%% post processing script for plotting
close all
clc
StartMat = mappedX(:, I(:,1))';
StopMat = mappedX(:, I(:, 2))';
figure
%arrow(StartMat, StopMat, 'LineWidth', 0.1);

subtrMat = StartMat - StopMat;
dist = sum(subtrMat, 2).^2;
qlimit = quantile(dist, 1.1);
idx = find(dist < qlimit);
NewStartMat = StartMat(idx, :);
NewStopMat = StopMat(idx, :);
arrow(NewStartMat, NewStopMat, 'LineWidth', 0.01);
Plotcluster(mappedX, categorical(webpage_classnames));

%%
legendstrings = categories(categorical(webpage_classnames));
%print('Cora_marge5e-3_softcauchy_arrows', '-depsc');

% ydata = tsne(X);
yStartMat = ydata(I(:,1), :);
yStopMat = ydata(I(:,2), :);
figure
ysubtrMat = yStartMat - yStopMat;
ydist = sum(ysubtrMat, 2).^2;
yqlimit = quantile(ydist, 0.9);
yidx = find(ydist < yqlimit);
yNewStartMat = yStartMat(yidx, :);
yNewStopMat = yStopMat(yidx, :);
arrow(yNewStartMat, yNewStopMat, 'LineWidth', 0.01);
Plotcluster(ydata', categorical(webpage_classnames));