function [hl, hp, x, m, sem] = plot_sem(x, y, color, linewidth, facealpha, smoothing, with_valid)
% plot line + patch (usually s.e.m.)
% returns handle to line and patch object

% if the flag with_valid is set, only the valid part of hte curve is
% plotted (due to smoothing the first and last part have to be removed)

if nargin < 3
    color = 'k';
end
if nargin < 4
    linewidth = 2;
end
if nargin < 5
    facealpha = 0.1;
end
if nargin < 6
    smoothing = 1;
end
if nargin < 7
    with_valid = false;
end

if numel(x) ~= size(y,2)
    error('dimensions do not match')
end

hl = NaN;
hp = NaN;
if ~isempty(y)
    if any(isnan(y(:)))
        warning('ignoring NaNs')
    end
    m = nanmean(y,1);
    sem = nanstd(bootstrp(1000,@nanmean,y));
    
    % smooth the data
    m = convn(m,ones(1,smoothing)/smoothing,'same');
    sem = convn(sem,ones(1,smoothing)/smoothing,'same');
    
    if with_valid
        % only plot valid part after smoothing
        delta = floor(smoothing / 2);
        x = x(delta:end-delta);
        m = m(delta:end-delta);
        sem = sem(delta:end-delta);
    end
    
    [hl, hp] = errorpatch(x, m, sem, color, linewidth, facealpha);
end
