function show_RT_vs_kernel_shape(sub_data, fig_format)
%SHOW_RT_VS_KERNEL_SHAPE  Show relationship between the shape of the PK and
%the average response times
%
%
%   SHOW_RT_VS_KERNEL_SHAPE() shows some figures

if nargin < 2
    % default: do not save figures
    fig_format = {};
end


% zscore reaction times
avg_rt = sub_data.c_mean;
datasets = unique(sub_data.dataset);
for i_data = 1:numel(datasets)
    avg_rt(sub_data.dataset == datasets(i_data)) = zscore(avg_rt(sub_data.dataset == datasets(i_data)));
end


figure('name','RT_vs_kernel_shape')

scatterplot(sub_data.PK_slope,avg_rt);
xlabel('PK slope')
ylabel('normalized RT')

figsave(gcf,'figs/Fig_6D',fig_format);
