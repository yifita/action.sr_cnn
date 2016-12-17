function [cls_results_avg, map_cate] = plot_category_accuracy(dataset, cls_results, savedir, draw)
load('output/categories.mat');
% channels = fieldnames(cls_results{1});
channels = {'person','scene','final','baseline'};
num_channels = length(channels);
map_cate = zeros(length(categories.names), num_channels);
for i = 1:num_channels
    cls_results_avg.(channels{i}).accuracy = cls_results(1).(channels{i}).accuracy;
    for j = 2:length(cls_results)
        cls_results_avg.(channels{i}).accuracy = cls_results(j).(channels{i}).accuracy + cls_results_avg.(channels{i}).accuracy;
    end
    cls_results_avg.(channels{i}).accuracy = cls_results_avg.(channels{i}).accuracy/length(cls_results);
    for c = 1:length(categories.names)
        target_cls = categories.classes_of(c);
        map_cate(c, i) = mean(cls_results_avg.(channels{i}).accuracy(target_cls));
    end
end
if exist('draw','var') && draw
    screensize = get( groot, 'Screensize' );
    for c = 1:length(categories.names)
        target_cls = categories.classes_of(c);
        xLabel = categories.names{c};
        % mAP plot
        colors = num2cell(hsv(num_channels),2);
        f = figure('Position',[0 0 screensize(3:4)]);
        if exist('categories', 'var')
            figurename = sprintf('Accuracy_%s.fig', categories.names{c});
        else
            figurename = 'Accuracy.fig';
        end
        for i = 1:num_channels
            if ~isempty(strfind(channels{i},'baseline'))
                plot(cls_results_avg.(channels{i}).accuracy(target_cls),'Color', colors{i});
            else
                plot(cls_results_avg.(channels{i}).accuracy(target_cls), 'Color', colors{i});
            end
            hold on;
        end
        for i = 1:num_channels
            hline = refline([0 mean(cls_results_avg.(channels{i}).accuracy(target_cls))]);
            hline.Color = colors{i};
            hline.LineStyle = ':';
        end
        legends = cell(num_channels*2,1);
        legends(1:num_channels) = channels;
        legends(num_channels+1:end) = cellfun(@(x) strcat('avg ', x), channels, 'uni', false);
        legend(legends,'Interpreter','none');
        set(gca,'XLim', [1 length(target_cls)], ...
            'XTick',1:length(target_cls),'XTickLabel', dataset.classes(target_cls),'XTickLabelRotation',45);
        xlabel(xLabel);
        %     title(sprintf('%s %d person, %d object, %d scene', merge, n_person, use_objects, use_scene),'Interpreter','none');
        hold off;
        savefig(f, fullfile(savedir, figurename))
    end
    f = figure('Position',[0 0 screensize(3:4)]);
    plot(map_cate);
    legend(categories.names, 'Interpreter','none');
    set(gca, 'XLim', [1 length(categories.names)], 'XTick', 1:length(categories.names), 'XTickLabel', categories.names,'XTickLabelRotation',45);
    savefig(f, fullfile(savedir, 'All_Categories_Accuracy.fig'));
end