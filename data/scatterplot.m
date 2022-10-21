function [r,p] = scatterplot(x,y,varargin)
%SCATTERPLOT Scatterplot with correlation coefficient/bubble plot.

% remark: perhaps play with 'markerfacealpha' (when having many points)

opt = struct(varargin{:});
if ~isfield(opt,'color')
    opt.color = 'w';
end
if ~isfield(opt,'marker')
    opt.marker = 'o';
end
if ~isfield(opt,'markerfacecolor')
    opt.markerfacecolor = 'b';
end
if ~isfield(opt,'markersize')
    opt.markersize = 5;
end
if ~isfield(opt,'linestyle')
    opt.linestyle = 'none';
end
if ~isfield(opt,'corrtype')
    corrtype = 'Pearson';
else
    corrtype = opt.corrtype;
    opt = rmfield(opt,'corrtype');
end
if ~isfield(opt,'axistype')
    axistype = 'normal';
else
    axistype = 'equal';
    opt = rmfield(opt,'axistype');
end


x = x(:);
y = y(:);

plot(x, y, opt)
hold on
box off

switch axistype
    case 'equal'
        % we want equal axis
        lim_max = max([xlim,ylim]);
        lim_min = min([xlim,ylim]);
        axis([lim_min lim_max lim_min lim_max])
        axis square
    otherwise
end

% % if we want a diagonal line
% plot(xlim,ylim,'k--')

if ~strcmp(corrtype,'none')
    [r,p] = corr(x,y,'type',corrtype);
    switch lower(corrtype)
        case 'pearson'
            corr_symbol = 'r';
        case 'spearman'
            corr_symbol = '\rho';
    end
    % title(sprintf('r = %2.2f, P = %e',r,p))
    
    if p > 0.001
        text(min(xlim) + 0.75*diff(xlim), min(ylim) + 0.06*diff(ylim), ...
            sprintf('%s = %2.2f\nP = %1.3f',corr_symbol,r,p),'Fontsize',13);
    else
        text(min(xlim) + 0.75*diff(xlim), min(ylim) + 0.06*diff(ylim), ...
            sprintf('%s = %2.2f\nP = %1.2e',corr_symbol,r,p),'Fontsize',13);
    end
else
    r = NaN;
    p = NaN;
end

box off
