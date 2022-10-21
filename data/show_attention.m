function show_attention(sub_data, fig_format)
%SHOW_ATTENTION  Plot PK slopes for focused and divided attention condition
%
%   SHOW_ATTENTION()
%
%   to save figures, specify fig_format = {'fig','png'}

if nargin < 2
    % default: do not save figures
    fig_format = {};
end

figure('name','attention_PK_slope','position',[194, 289, 725, 259])
tiledlayout(1,2)

nexttile; hold on
PK_foc = sub_data.bw(sub_data.dataset == 3,:);
PK_div = sub_data.bw(sub_data.dataset == 5,:);
xlim([0.5,8.5]);
ylim([0,0.8]);
plot(mean(PK_foc),'linewidth',2)
plot(mean(PK_div),'linewidth',2)
xlabel('Frame number');
ylabel('Stimulus impact');
legend('Focused','Divided','location','northwest')
legend(gca,'boxoff')

nexttile;
axis([ -0.3, 0.5, -0.1, 0.5])
axis square
hold on
plot([0 0],ylim,'k--')
plot(xlim,[0 0],'k--')
plot(xlim,ylim,'k--')
plot(sub_data.PK_slope(sub_data.dataset == 3),sub_data.PK_slope(sub_data.dataset == 5), ...
    'Marker','o','MarkerFaceColor','k','LineStyle','none','Color','k');
xlabel('PK slope (focused)')
ylabel('PK slope in (divided)')

figsave(gcf,'figs/Fig_S9B-C',fig_format);
