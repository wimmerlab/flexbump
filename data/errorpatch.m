function [hl, hp] = errorpatch(t, m, sem, color, linewidth, facealpha)
% plot line + patch (usually s.e.m.)
% returns handle to line and patch object

if nargin < 4
    color = 'k';
end
if nargin < 5
    linewidth = 2;
end
if nargin < 6
    facealpha = 0.1;
end

t = t(:);
m = m(:)';

if size(sem,1) > 1 && size(sem,2) > 1
    % lower and upper errorbars are different
    hp = patch([t' t(end:-1:1)' t(1)],[sem(1,:) sem(2,end:-1:1) sem(1,1)],color,'edgecolor','none','facealpha',facealpha);
else
    % standard error (equal in both directions)
    sem=sem(:)';
    hp = patch([t' t(end:-1:1)' t(1)],[m+sem m(end:-1:1)-sem(end:-1:1) m(1)+sem(1)],color,'edgecolor','none','facealpha',facealpha);
end

hold on
hl(1) = plot(t, m, 'color',color,'linewidth',linewidth);

