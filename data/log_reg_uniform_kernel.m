function [beta, AIC] = log_reg_uniform_kernel(x,y)
%LOG_REG  Perform logistic regression under the contraint of a constant
%temporal kernel
%
%   LOG_REG()

x = [ones(size(x,1),1), x];

% % unconstrained logistic regression, gives the same result and glmfit:
% beta = fminsearch(@(beta) -LL(beta,x,y),0.5 * ones(9,1));

% contraint logistic regression (linear kernel)
[beta, max_neg_LL] = fminsearch(@(beta) -LL_linear_kernel(beta,x,y),0.05 * ones(2,1));

% Akaike information criterion
AIC = 2 * numel(beta) + 2 * max_neg_LL;

end

% cumulative Gaussian (probit regression)
function y = f(x)
y = normcdf(x,0,1);
end

% log-likelihood function for weights following a constant function
function ll = LL_linear_kernel(b,x,y)

beta = [b(1); b(2) * ones(8,1)];

ll = y'*log(f(x*beta)) + (1-y)'*log(1-f(x*beta));
end
