function figsave (handle, filename, figformat)
% save figure in the specified formats
% 'format' can be a cell array of different formats => mutliple files are
% saved

if isempty(figformat)
    return
end

for i=1:numel(figformat)
    switch (figformat{i})
        case 'png'
            %print(handle, '-dpng', '-r300', [filename,'.png']);
            saveas(handle, [filename '.png'], 'png');
        case 'fig'
            saveas(handle, [filename '.fig'], 'fig');
        otherwise
            error('unknown format')
    end
end
