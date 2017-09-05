% ignore the diagonal entries of A
function [eta, nu] = eta_nu_fmin2(dp, A, init, options)

if nargin <=3
    options = optimset('GradObj', 'on','Hessian','on');
end


dpp = diag(1./sum(dp,2))*dp;
c = dpp*dpp';  % c is the D by D matrix of inner products
D = size(c, 1);
c = c(find(~(mod([1:D^2],D+1)==1)));
A = A(find(~(mod([1:D^2],D+1)==1)));
c1 = c(find(A == 1));
c0 = c(find(A == 0));
%disp(size(c1)), disp(size(c0))
init(2)  = min([min(-init(1)*c(:)) - 0.1, init(2)]); % make sure probability is in [0,1];

theta_opt = fminunc(@(theta)likelihood(theta, c1, c0), init, options);
eta = theta_opt(1); 
nu  = theta_opt(2);


function  [f, grad, hansen] = likelihood(theta, c1, c0)
DD = (length(c1(:)) + length(c0(:)))/2;
eta = theta(1); nu = theta(2);
a = sum(c1(:));  b = length(c1(:));
%f = eta*sum(c1(:)) + nu*length(c1(:)) + sum(log(1- exp(eta*c0(:) + nu)));
expVal1 = 1./(1 - exp(eta*c0(:) + nu));
f = eta*a + nu*b -  sum(log(expVal1));
f = -f/DD; 

if nargout >=2
      grad = zeros(2,1);
      expVal1 =1 - expVal1;
      expValc = expVal1.*c0(:);
      grad(1) = a + sum(expValc);
      grad(2) = b + sum(expVal1);
      grad = -grad/DD;
end

if nargout >=3
      hansen = zeros(2,2);
      hansen(1,1) = sum(expValc.*(c0(:) - expValc));
      hansen(1,2) = sum(expValc.*(1 - expVal1));
      hansen(2,1) = hansen(1,2);
      hansen(2,2) = sum(expVal1.*(1 - expVal1)); 
      hansen = -hansen/DD;
end

