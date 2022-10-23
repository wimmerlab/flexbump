function [trl_data, sub_data] = analysis_logreg(trl_data, sub_data)
%ANALYSIS_LOGREG  Perform logistic regression on data from cardinal/diagonal experiments
%
%   ANALYSIS_LOGREG() performs logistic regression on individual
%   subjects and analysis the heterogeneity in temporal weighting with the
%   primacy-recency index


nsub = size(sub_data,1);

sub_data.bw = nan(nsub,8);                  % stimulus weights
sub_data.bias = nan(nsub,1);                 % bias
sub_data.bw_se = nan(nsub,8);               % standard error of weights
sub_data.bw_lin = nan(nsub,8);              % stimulus weights (constraint: linear kernel)
sub_data.bw_slope = nan(nsub,1);            % corresponding slope
sub_data.AIC = nan(nsub,1);                 % AIC for linear kernel
sub_data.AIC_uniform = nan(nsub,1);         % AIC for constant (uniform) kernel

% predicted respones by the regression models
n_trials = size(trl_data,1);
trl_data.yfit = NaN(n_trials,1);            % prediction of subject's response by regression model
trl_data.yfit_lin = NaN(n_trials,1);        % prediction of subject's response by regression model (constraint: linear kernel)
trl_data.yfit_cont = NaN(n_trials,1);       % prediction of subject's response by regression model

for isub = 1:nsub

    % get subject-wise data
    itrl = trl_data.dataset == sub_data.dataset(isub) & trl_data.sub == sub_data.sub(isub);

    % perform regression analyses

    % 1/ psychophysical kernel - logistic regression
    [b,~,stats] = glmfit(trl_data.dv(itrl,:),trl_data.resp(itrl) == 1,'binomial','link','probit');
    sub_data.bw(isub,:) = b(2:end);
    sub_data.bw_se(isub,:) = stats.se(2:end);
    sub_data.bias(isub) = b(1);
    trl_data.yfit(itrl) = glmval(b, trl_data.dv(itrl,:), 'probit');

    % 2/ logistic regression, constraining to a linear weight kernel
    [beta,AIC] = log_reg_lin_kernel(trl_data.dv(itrl,:),trl_data.resp(itrl) == 1);
    sub_data.bw_lin(isub,:) = beta(2)+(0:7)'*beta(3);
    sub_data.bw_slope(isub) = beta(3);
    trl_data.yfit_lin(itrl) = glmval([beta(1) sub_data.bw_lin(isub,:)].', trl_data.dv(itrl,:), 'probit');

    % 3/ AIC for linear kernel and for uniform kernel
    [~,AIC_uniform] = log_reg_uniform_kernel(trl_data.dv(itrl,:),trl_data.resp(itrl) == 1);
    sub_data.AIC(isub) = AIC;
    sub_data.AIC_uniform(isub) = AIC_uniform;

end

% compute normalized PK slope
sub_data.PK_slope = sub_data.bw_slope ./ mean(abs(sub_data.bw_lin),2);

% classify subjects into "primacy", "uniform" and "recency" groups, based
% on the AIC
sign_recency = sub_data.AIC < sub_data.AIC_uniform & sub_data.bw_slope > 0;           % subjects with significant recency
sign_primacy = sub_data.AIC < sub_data.AIC_uniform & sub_data.bw_slope < 0;           % subjects with significant primacy
PK_type(1:size(sub_data,1)) = {'uniform'};
PK_type(sign_recency) = {'recency'};
PK_type(sign_primacy) = {'primacy'};
sub_data.PK_type = categorical(PK_type(:));
