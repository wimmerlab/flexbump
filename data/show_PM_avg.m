function show_PM_avg(sub_data, fig_format)
%SHOW_PM_AVG  Plot psychometric and chronometric curves for
%all datasets combined (summary plot)
%
%   SHOW_PM_AVG()
%
%   to save figures, specify fig_format = {'fig','png'}

if nargin < 2
    % default: do not save figures
    fig_format = {};
end

% --- average psychometric and chronometric curve across subjects for each of the datasets
figure('name',sprintf('Average psychometric and chronometric curves'),'position',[450 84 450 716]);

idx = 1:size(sub_data,1);

subplot(2,1,1); hold on

% plot average psychometric curve (data points) across subjects
errorbar(mean(sub_data.x(idx,:)), mean(sub_data.p(idx,:)), ...
    std(bootstrp(1000,@mean,sub_data.p(idx,:))),...
    'marker','s','MarkerFaceColor','k','color','k','linewidth',2,'markersize',8);
plot(mean(sub_data.x(idx,:)), mean(sub_data.p(idx,:)),'k','linewidth',2)

xlabel('Category-level average')
ylabel('P(cardinal)')
axis([-0.52 0.52 0 1])
plot(xlim,[0.5 0.5],'k--');
plot([0 0],ylim,'k--');

subplot(2,1,2); hold on

% correct trials
h(1) = errorbar(mean(sub_data.x(idx,:)), 1000 * mean(sub_data.c_corr(idx,:)), ...
    1000 * std(bootstrp(1000,@mean,sub_data.c_corr(idx,:))),...
    'marker','s','MarkerFaceColor','k','color','k','linewidth',2,'markersize',8);

% incorrect trials
h(2) = errorbar(mean(sub_data.x(idx,:)), 1000 * nanmean(sub_data.c_incorr(idx,:)), ...
    1000 * std(bootstrp(1000,@nanmean,sub_data.c_incorr(idx,:))),...
    'marker','s','MarkerFaceColor','k','color','k','linewidth',2,'markersize',8,'linestyle','--');

xlabel('Category-level average')
ylabel('Avg. RT - <RT> (ms)')
axis([-0.52 0.52 -50 120])
plot([0 0],ylim,'k--');

legend(h,'correct trials','error trials');

figsave(gcf,sprintf('figs/Fig_S10D'),fig_format);

