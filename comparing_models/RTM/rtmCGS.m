function[] = rtmCGS(dataset,T_str, seed_str, exactFlag_str, percentage_str, negativePercentage_str, paramOn_str)
% function[] = rtmCGS(dataset,T_str, seed_str, exactFlag_str, percentage_str, negativePercentage_str, paramOn_str)
%
% Runs Chang's "Relational Topic Model" on networks which have text on the nodes.
% Note it is helpful to read the little pdf note on the web site.
% Also note that the inputs above are actually in string format (since this is
% useful when using the Matlab compiler).
% 
% INPUTS:
%    dataset: 'e' for ENRON, 'n' for Netflix
%    T: # of topics
%    seed: random seed
%    exactFlag: 0 for approximate treatment of non-edges as detailed in Arthur's writeup
%               1 for exact treatment of non-edges
%               2 for "exact" treatment of non-edges but only doing it once per document per sweep
%               3 to just run LDA and not RTM
%    percentage: If exactFlag = 1,2 this percentage will determine the percentage of non-edges to sub-sample
%    negativePercentage: determines the influence of the non-edge term to the sampler.
%			 1 means "model all 0's in adjacency matrix as observed non-edges".
%			 0 means  "model all 0's in adjacency matrix as missing data"
%			 0.5 means "model 50% of 0's in adjacency matrix as observed non-edges".
%    paramOn_str: Select version of parameter estimation (1 or 2, see below).
%
% For questions please email Arthur

T = str2num(T_str);
seed = str2num(seed_str);
exactFlag = str2num(exactFlag_str);
percentage = str2double(percentage_str);
negativePercentage = str2double(negativePercentage_str);
paramOn = str2num(paramOn_str);

%--------------------------------------------
% parameters
%--------------------------------------------
P=1; % one processor

ITER  = 200;
rand('state',sum(100*clock))

% alpha priors
ga=2;
gb=4;

% beta priors
gc=2;
gd=4;

alpha=0.1;
beta=0.1;

%eta = rand(T,1);
scalareta = 1;
eta = ones(T,1) * scalareta;
%eta = eta ./ sum(eta);
%nu = - sum(eta);
nu = - 5;

%-------------------------------------------- 
% read corpus 
%-------------------------------------------- 
if (dataset=='n')
  load datasets/docword.netflix.train.mat
  % The adjacency matrix between the nodes in the graph
  load datasets/A.netflixCombined.train.mat
  A = full(Atrain);
  A = double(A);
elseif (dataset=='e')
  load datasets/docword.enron.train.mat
  load datasets/A.enron.train.mat
  A = full(Atrain);
  A = double(A);
elseif (dataset=='m')
  load datasets/docword.nips.train.mat
  % Random adjacency (this is just a toy example)
  A = rand(D,D);
  A = (A + A') ./ 2;
  A = A < 0.2;
end


%--------------------------------------------
% random initial assignment
%--------------------------------------------
z  = floor(T*rand(N,1)) + 1;
% count matrices
wp = zeros(W,T);
dp = zeros(D,T);
for n = 1:N
    wp(w(n),z(n)) = wp(w(n),z(n)) + 1;
    dp(d(n),z(n)) = dp(d(n),z(n)) + 1;
end
ztot = sum(wp,1);


Nd = sum(dp,2);
zeroIndex = find (Nd==0);

% noninformative hack
dp(zeroIndex,:) = ones(length(zeroIndex),T);

Nd = sum(dp,2);

%Initialization 
scalareta = 0.5;
eta = ones(T,1) * scalareta;

dp2 = dp ./ repmat(Nd,1,T);
c = (repmat(eta',D,1) .* dp2)*(dp2)';  % c is the D by D matrix of inner products
c = c(find(~(mod([1:D^2],D+1)==1)));

%nu  = min([min(-max(eta)*c(:)) - 1]);

nu = -max(c(:)) - 5;
theta = [eta; nu];

tic;

%--------------------------------------------
% iterate
%--------------------------------------------
for iter = 1:ITER

  fprintf('iter %d; dataset=%s, T=%d, seed=%d, approx=%d, subsample=%d, negative=%d, param=%d \n', iter, dataset, T, seed, exactFlag, percentage, negativePercentage, paramOn);

  ztot = sum(wp);  
  zPrev = z;

  % The Gibbs sampling is done within the C mex file    
  [z, wp, dp] = RTMmex(z, wp, dp, ztot, w, d, alpha, beta, Nd, A, eta, nu, exactFlag, percentage, negativePercentage);

  % print topics
  if (mod(iter,10)==0)
    ztot = sum(wp);
    for t=1:T
      [xsort,isort] = sort(-wp(:,t));
      fprintf('[t%d] (%.3f) ', t, ztot(t)/N);
      for i=1:min(8,W)
        fprintf('%s ', word{isort(i)});
      end
      fprintf('\n');
    end  
  end

  % Parameter estimation.  I disable it sometimes because it is sometimes ill-conditioned
  if (iter > 5 && paramOn == 1)
    [scalareta,nu] = eta_nu_fmin2(dp, A, [scalareta,nu]');
    scalareta
    nu
    eta = ones(T,1) * scalareta;
    %[eta,nu] = eta_nu_Vec(dp, A, [eta;nu]');
  end

  % vector version
  if (iter > 15 && paramOn == 2)
    theta = [eta;nu];
    dpp = diag(1./sum(dp,2))*dp; %normize the counting matrix
    %stepSize = 0.1;  Numiter =  1*1e5; Diter = Numiter/100;  Burnin = 10; 
    %record the solution every Diter steps, and the final value  is the average of the recorded solutions after Burnin steps.   
    %now = clock;
    %[theta, theta_last, etaMean, nu, f] = mex_stograd(dpp, A, theta, stepSize, Numiter, Diter,Burnin);
    %theta',  etime(clock, now)

    step =0.001;  Numiter = 25;  Tol =  0.00001*D*(D-1);
    [theta, grad, f] = mex_grad(dpp, A, theta, step, Numiter, Tol);

    eta=theta(1:T);
    nu = theta(T+1);
  end
  
  % Right now I disable the parameter estimation (Minka's update) of alpha, beta
  if (0) % iter > 5)

    tolerance = 0.0001;

    % Alpha Update 
    Nd = sum(dp,2);
    for subiter=1:5
      num = sum(sum(digamma(alpha + dp) - digamma(alpha)));
      den = sum(digamma( T*alpha + Nd) - digamma(T*alpha));
      alphaTemp = alpha;
      alpha = (ga - 1 + alpha * num) / (gb + T*den);
      if (abs(alphaTemp - alpha) < tolerance)
        %fprintf('break in alpha %d\n',subiter);
        break;
      end
    end % end subiter

    % Beta Update 
    ztot = sum(wp);
    for subiter=1:5
      num = sum(sum(digamma(beta + wp) - digamma(beta)));
      den = sum(digamma( W*beta + ztot) - digamma(W*beta));
      betaTemp = beta;
      beta = (gc - 1 + beta * num) / (gd + W*den);
      if (abs(betaTemp - beta) < tolerance)
        %fprintf('break in beta %d\n',subiter);
        break;
      end
    end % end subiter
    
    [alpha,beta]
  
  end

end

elapsedTime = toc;

