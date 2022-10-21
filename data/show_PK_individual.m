function show_PK_individual(sub_data, fig_format)
%SHOW_PK_INDIVIDUAL  Show all individual psychophysical kernels in one
%figure, ordered by PK slope
%
%   SHOW_PK_INDIVIDUAL() show individual subject's PK

if nargin < 2
    % default: do not save figures
    fig_format = {};
end

% show kernels (logistic regression weights) for individual subjects

primacy_sub = find(sub_data.PK_type=='primacy');
recency_sub = find(sub_data.PK_type=='recency');
uniform_sub = find(sub_data.PK_type=='uniform');

[~,order_primacy] = sort(sub_data.PK_slope(primacy_sub));
[~,order_uniform] = sort(sub_data.PK_slope(uniform_sub));
[~,order_recency] = sort(sub_data.PK_slope(recency_sub));

subjects = [primacy_sub(order_primacy); uniform_sub(order_uniform); recency_sub(order_recency)];

figure('name','Individual PKs','position',[380   110   699   620]);
th = tiledlayout(7,9,'TileSpacing','none','Padding','none');

h = [];
for isub = 1:numel(subjects)

    my_sub = subjects(isub);

    nexttile; hold on

    errorbar(sub_data.bw(my_sub,:),sub_data.bw_se(my_sub,:),'color',[0.5 0.5 0.5], ...
        'linestyle','none','marker','.','capsize',4,'linewidth',0.75);
    p = plot(sub_data.bw_lin(my_sub,:),'-','linewidth',2);
    switch(sub_data.PK_type(my_sub))
        case 'primacy'
            set(p,'color','b');
            h(1) = p;
        case 'uniform'
            set(p,'color','k');
            h(2) = p;
        case 'recency'
            set(p,'color','r');
            h(3) = p;
    end

    axis([0 9 -0.5 1.8])
    plot(xlim,[0 0],'k:','linewidth',1);
    %     title({sprintf('Sub %d/%d:',sub_data.dataset(isub),sub_data.sub(isub)), sprintf('%s',sub_data.PK_type(isub))});
    if mod(isub,9) == 1
        yticks([0 1])
        set(gca,'xtick',[])
    else
        axis off
    end
    % set(gca,'xtick',[])
    % set(gca,'ytick',[])
end
xlabel(th,'Frame number','fontsize',14,'FontName','bodoni')
ylabel(th,'Stimulus impact','fontsize',14,'FontName','bodoni')

% histogram of PK slopes
nexttile([1,2]);
hold on
bins = -0.5:0.035:0.5;
histogram(sub_data.PK_slope(sub_data.PK_type == 'uniform'),bins,'facecolor','k')
histogram(sub_data.PK_slope(sub_data.PK_type == 'recency'),bins,'facecolor','r')
histogram(sub_data.PK_slope(sub_data.PK_type == 'primacy'),bins,'facecolor','b')
box off
xlabel('PK slope')
ylabel('Subjects')
axis([-0.5 0.5 0 12])
set(gca,'xtick',-1:.2:1);
set(gca,'ytick',0:2:10);
plot(mean(sub_data.PK_slope(sub_data.PK_type == 'uniform')),12,'kv','markerfacecolor','k')
plot(mean(sub_data.PK_slope(sub_data.PK_type == 'recency')),12,'kv','markerfacecolor','r')
plot(mean(sub_data.PK_slope(sub_data.PK_type == 'primacy')),12,'kv','markerfacecolor','b')

% nexttile;
% axis off
% l=legend(gca,h,'primacy','uniform','recency');
% set(l,'FontName','bodoni','fontsize',12)

figsave(gcf,'figs/Fig_S8',fig_format);

