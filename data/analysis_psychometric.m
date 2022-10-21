function [sub_data] = analysis_psychometric(trl_data, sub_data, fig_format, n_bins)
%ANALYSIS_PSYCHOMETRIC  Plot psychometric and chronometric curves for
%all subjects
%
%   ANALYSIS_PSYCHOMETRIC()
%
%   to save figures, specify fig_format = {'fig','png'}

nsub = size(sub_data,1);

if nargin < 3
    % default: do not save figures
    fig_format = {};
end

% --- bin stimulus evidence: in 6 bins with equal number of trials in each
% bin
% note: in dataset 2, we only have 5 bins (the 5 categories), with the zero
% category having twice the number of trials; the zero evidence trials are
% mapped here to bins 3 and 4 (using 6 bins)
if nargin < 4
    n_bins = 6;                 % use n_bins bins, with equal number of trials in each bin; was 6
end
assert(n_bins >= 5);        % for dataset 1, we need 5 bins
trl_data.stim_bin = nan(size(trl_data,1),1);

for isub = 1:nsub
    % get subject-wise data
    itrl = trl_data.dataset == sub_data.dataset(isub) & trl_data.sub == sub_data.sub(isub);

    if sub_data.dataset(isub) == 1
        % for dataset 1, we use the binning defined by the 5 discrete
        % categories (high/low evidence for orthogonal, high/low evdidence
        % for diagonal, and neutral (zero evidence)
        trl_data.stim_bin(itrl) = trl_data.dvmucat(itrl) + 3;
    else
        % define bins for the other subjects from dataset 2 and 3
        trl_data.stim_bin(itrl) = equalbins(trl_data.dvmu(itrl),n_bins);
    end
end

% --- make psychometric and chronometric curves for binned stimulus evidence
sub_data.x = nan(size(sub_data,1),n_bins);                      % x value of psychometric curve
sub_data.p = nan(size(sub_data,1),n_bins);                      % y value of psychometric curve
sub_data.c = nan(size(sub_data,1),n_bins);                      % y value of chronometric curve
sub_data.c_corr = nan(size(sub_data,1),n_bins);                 % y value of chronometric curve (correct trials)
sub_data.c_incorr = nan(size(sub_data,1),n_bins);               % y value of chronometric curve (incorrect trials)
sub_data.c_incorr_n_trials = nan(size(sub_data,1),n_bins);      % number of trials in each bin for incorrect responses
sub_data.c_mean = nan(size(sub_data,1),1);                      % mean response time for each subject

for isub = 1:nsub

    % get subject-wise data
    itrl = trl_data.dataset == sub_data.dataset(isub) & trl_data.sub == sub_data.sub(isub);

    % psychometric curve for the data
    stim_bins = unique(trl_data.stim_bin(itrl));
    for j=1:numel(stim_bins)
        sub_data.x(isub,j) = mean(trl_data.dvmu( itrl & trl_data.stim_bin == stim_bins(j) ));
        sub_data.p(isub,j) = mean(trl_data.resp( itrl & trl_data.stim_bin == stim_bins(j) ) == 1);
    end

    % chronometric curve for the data
    stim_bins = unique(trl_data.stim_bin(itrl));
    sub_data.c_mean(isub) = mean(trl_data.rt(itrl));
    for j=1:numel(stim_bins)
        sub_data.c(isub,j) = mean(trl_data.rt( itrl & trl_data.stim_bin == stim_bins(j) )) - sub_data.c_mean(isub);                                   % RT all trials
        sub_data.c_corr(isub,j) = mean(trl_data.rt( itrl & trl_data.iscor == 1 & trl_data.stim_bin == stim_bins(j) )) - sub_data.c_mean(isub);        % RT correct trials
        sub_data.c_incorr(isub,j) = mean(trl_data.rt( itrl & trl_data.iscor == 0 & trl_data.stim_bin == stim_bins(j) )) - sub_data.c_mean(isub);      % RT incorrect trials
        sub_data.c_incorr_n_trials(isub,j) = numel(trl_data.rt( itrl & trl_data.iscor == 0 & trl_data.stim_bin == stim_bins(j) ));                    % number of trials for incorrect responses
    end

end


end

