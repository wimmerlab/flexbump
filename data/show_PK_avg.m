function show_PK_avg(sub_data, fig_format)
%SHOW_PK_AVG  Show histogram of PK slopes
%
%   SHOW_PK_AVG() shows the average PKs for subjects classified as primacy,
%   uniform and recency 
%
%   to save figures, specify fig_format = {'fig','png'}

if nargin < 2
    % default: do not save figures
    fig_format = {};
end


figure('name','PRI and examples (weights not normalized)','position',[114, 383, 1230, 331]);
tiledlayout(1,3)

nexttile;
axis([0 9 -0.08 1]); hold on;
plot(1:8,sub_data.bw(sub_data.PK_type == 'primacy',:)','k','linewidth',0.1)
plot_sem(1:8,sub_data.bw(sub_data.PK_type == 'primacy',:),'b');
title(sprintf('primacy: %d / %d subjects', sum(sub_data.PK_type == 'primacy'), height(sub_data)));
xlabel('Frame number')
ylabel('Stimulus impact')
set(gca,'xtick',1:8)

nexttile;
axis([0 9 -0.08 1]); hold on;
plot(1:8,sub_data.bw(sub_data.PK_type == 'uniform',:)','k','linewidth',0.1)
plot_sem(1:8,sub_data.bw(sub_data.PK_type == 'uniform',:),'k');
title(sprintf('uniform: %d / %d subjects', sum(sub_data.PK_type == 'uniform'), height(sub_data)));
xlabel('Frame number')
ylabel('Stimulus impact')
set(gca,'xtick',1:8)

nexttile;
axis([0 9 -0.08 1]); hold on;
plot(1:8,sub_data.bw(sub_data.PK_type == 'recency',:)','k','linewidth',0.1)
plot_sem(1:8,sub_data.bw(sub_data.PK_type == 'recency',:),'r');
title(sprintf('recency: %d / %d subjects', sum(sub_data.PK_type == 'recency'), height(sub_data)));
xlabel('Frame number')
ylabel('Stimulus impact')
set(gca,'xtick',1:8)

figsave(gcf,'figs/Fig_5B',fig_format);
