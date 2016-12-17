function cls_results = visualize_cls_res(dataset, output_cache, categories, draw, varargin)
ip = inputParser;
ip.addParameter('baseline', 'output/srcnn_cachedir/conv_ucf_0_p_0_c_scene', @ischar);
ip.addParameter('weights', [], @ismatrix);
ip.addParameter('split', 1,   @isscalar);
ip.addParameter('set', 'test', @ischar);
ip.parse(varargin{:});
opts = ip.Results;
if ~exist('draw', 'var')
    draw = true;
end
if isempty(categories) && strcmp(dataset.name,'ucf')
    load('output/categories');
    num_categories = length(categories.names);
else
    num_categories = 1;
end

%% parse setting from given folder name
split = opts.split;
try 
    ld = load(output_cache);
    parsed = regexp(output_cache, sprintf('(flow)?.*%s_(\\d)_p_(x|0)_c(_scene)?_?([^/]*)',dataset.name),'tokens');
    savedir = fileparts(output_cache);
catch
    ld = load(fullfile(output_cache, sprintf('split_%d', split), 'cls_res'));
    parsed = regexp(output_cache, sprintf('(flow)?.*%s_(\\d)_p_(x|0)_c(_scene)?_?([^/]*)',dataset.name),'tokens');
    savedir = fullfile(output_cache, sprintf('split_%d', split));
end
assert(~isempty(parsed{1}), 'Cannot parse file name');
switch opts.set
    case 'test'
        gt_cls = dataset.video_cls(dataset.test_splits{split});
    case 'val'
        gt_cls = dataset.video_cls(dataset.val_splits{split});
    otherwise
        error('invalid set option')  
end
clsName_to_id = containers.Map(dataset.classes, 1:length(dataset.classes));
gt_cls_ids = cell2mat(clsName_to_id.values(gt_cls));
gt_cls_ids = reshape(gt_cls_ids, 1, []);

% baseline cls_res.mat file
if ~isempty(parsed{1}{1})
    opts.baseline = sprintf('output/srcnn_cachedir/conv_flow_%s_0_p_0_c_scene/split_%d/orig_cls_res.mat',dataset.name, opts.split);
else
    opts.baseline = sprintf('output/srcnn_cachedir/conv_%s_0_p_0_c_scene/split_%d/cls_res.mat',dataset.name, opts.split);
end
if ~isempty(parsed{1}{2})
    n_person = str2num(parsed{1}{2});
end
use_objects = ~isempty(parsed{1}) && strcmp(parsed{1}{3}, 'x');
    
use_scene = ~isempty(parsed{1}{4});
merge = '';
if ~isempty(parsed{1}) && ~isempty(parsed{1}{5})
    merge = parsed{1}{5};
end

% build hashmap of clsName_testIdx
test_cls_name = dataset.video_cls((dataset.test_splits{split}));
clsName_2_testIdx = containers.Map;
for i = 1:length(dataset.classes)
    clsName_2_testIdx(dataset.classes{i}) = find(strcmp(test_cls_name, dataset.classes{i}));
end

cls_results = struct();
if (n_person>0)
    prob = softmax(ld.person_cls);
    [~, pred_cls_id] = max(prob);
    cls_results.person.pred_cls_id = pred_cls_id;
    cls_results.person.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.person.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
    cls_results.person.statics = score_statics(ld.person_cls);
end
if use_objects
    prob = softmax(ld.obj_cls);
    [~, pred_cls_id] = max(prob);
    cls_results.objects.pred_cls_id = pred_cls_id;
    cls_results.objects.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.objects.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
    cls_results.objects.statics = score_statics(ld.obj_cls);
end
if use_scene
    prob = softmax(ld.scene_cls);
    [~, pred_cls_id] = max(prob);
    cls_results.scene.pred_cls_id = pred_cls_id;
    cls_results.scene.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.scene.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
    cls_results.scene.statics = score_statics(ld.scene_cls);
end
if sum([n_person>0 use_scene])>1
    score = softmax(ld.person_cls) + softmax(ld.scene_cls);
    [~, pred_cls_id] = max(score);
    cls_results.person_scene.pred_cls_id = pred_cls_id;
    cls_results.person_scene.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.person_scene.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
    cls_results.person_scene.statics = score_statics(score);
end
if sum([n_person>0 use_scene])>1
    score = max(ld.person_cls, ld.scene_cls);
    merged_prob = softmax(score);
    [~, pred_cls_id] = max(merged_prob);
    cls_results.max_of_person_scene.pred_cls_id = pred_cls_id;
    cls_results.max_of_person_scene.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.max_of_person_scene.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
    cls_results.max_of_person_scene.statics = score_statics(score);
end
if sum([n_person>0 use_scene use_objects])>1
    merged_prob = softmax(ld.merged_cls);
    [~, pred_cls_id] = max(merged_prob);
    cls_results.merged.pred_cls_id = pred_cls_id;
    cls_results.merged.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.merged.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
    cls_results.merged.statics = score_statics(ld.merged_cls);
end
if isfield(ld, 'weighted_cls') || ~isempty(opts.weights)
    if  isempty(opts.weights)
        if ~iscell(ld.weighted_cls)
            weighted_cls = {ld.weighted_cls};
        end
    else
        if ~iscell(opts.weights)
            opts.weights = {opts.weights};
        end
        person_cls = []; object_cls = []; scene_cls = [];
        if n_person > 0
            person_cls = ld.person_cls;
        end
        if use_objects
            object_cls = ld.object_cls;
        end
        if use_scene
            scene_cls = ld.scene_cls;
        end
        weighted_cls = cellfun(@(x) x' * [person_cls; scene_cls; object_cls], opts.weights, 'uni',false);
    end
    weighted_fields = arrayfun(@(i) sprintf('weighted_%d',i), 1:length(weighted_cls), 'uni', false);
    for i = 1:length(weighted_cls)
        weighted_prob = softmax(weighted_cls{i});
        [~, pred_cls_id] = max(weighted_prob);
        cls_results.(weighted_fields{i}).pred_cls_id = pred_cls_id;
        cls_results.(weighted_fields{i}).accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
        cls_results.(weighted_fields{i}).confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
        cls_results.(weighted_fields{i}).statics = score_statics(ld.merged_cls);
    end
end
[~, pred_cls_id] = max(ld.prob);
cls_results.final.pred_cls_id = pred_cls_id;
cls_results.final.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
cls_results.final.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);

% baseline
clear ld
try
    ld = load(opts.baseline);
    [~, pred_cls_id] = max(ld.prob);
    cls_results.baseline.pred_cls_id = pred_cls_id;
    cls_results.baseline.accuracy = cls_accuracy(pred_cls_id, gt_cls_ids);
    cls_results.baseline.confusion_matrix = cls_confusion(pred_cls_id, gt_cls_ids);
catch
end

channels = fieldnames(cls_results);
for i = 1:length(channels)
    cls_results.(channels{i}).confusion = cellfun(@(x) list_confusion_class(dataset, ...
        cls_results.(channels{i}).confusion_matrix, x), dataset.classes, 'uni',false);
end
%% visualize
if ~draw
    return;
end
for c = 1:num_categories
    % mAP plot
    channels = fieldnames(cls_results);
    num_channels = length(channels);
    screensize = get( groot, 'Screensize' );
    colors = num2cell(hsv(num_channels),2);
    if ~isempty(categories)
        target_cls = categories.classes_of(c);
        xLabel = categories.names{c};
    else
        target_cls = 1:length(dataset.classes);
        xLabel = '';
    end

    f = figure('Position',[0 0 screensize(3:4)]);
    if exist('categories', 'var')
        figurename = sprintf('Accuracy_%s.fig', xLabel);
    else
        figurename = 'Accuracy.fig';
    end
    for i = 1:num_channels
        if ~isempty(strfind(channels{i},'baseline'))
            plot(cls_results.(channels{i}).accuracy(target_cls), '--');
        else
            plot(cls_results.(channels{i}).accuracy(target_cls), 'Color', colors{i});
        end
        hold on;
    end
    legend(fieldnames(cls_results),'Interpreter','none');
    set(gca,'XLim', [1 length(target_cls)], 'YLim', [0, 1], ...
        'XTick',1:length(target_cls),'XTickLabel', dataset.classes(target_cls),'XTickLabelRotation',45);
    xlabel(xLabel);
    title(sprintf('%s %d person, %d object, %d scene', merge, n_person, use_objects, use_scene),'Interpreter','none');
    hold off;
    savefig(f, fullfile(savedir, figurename))

    % hitmap plot
    for i = 1:num_channels
        if exist('categories', 'var')
            figurename = sprintf('Confusion_%s_%s.fig',  channels{i},  xLabel);
        else
            figurename = sprintf('Confusion_%s.fig', channels{i});
        end
        f = figure('Position',[0 0 screensize(3:4)]);
        colormap('hot');
        imagesc(cls_results.(channels{i}).confusion_matrix(target_cls, :)');
        title(sprintf('%s channel: %s', merge, channels{i}),'Interpreter','none');
        colorbar
        set(gca, 'YDir', 'normal', 'YLim', [1 length(dataset.classes)], ...
            'XTick', 1:length(dataset.classes(target_cls)), 'YTick', 1:length(dataset.classes),...
            'XTickLabel', dataset.classes(target_cls), 'YTickLabel', dataset.classes, 'XTickLabelRotation',45, 'YTickLabelRotation',45);
        xlabel(xLabel);
        savefig(f,fullfile(savedir, figurename));
        close all;
    end
end
%% class accuracy
    function accuracy = cls_accuracy(pred, gt)
        accuracy = nan(length(dataset.classes),1);
        for j = 1:length(dataset.classes)
            pos = (gt == j);
            true_pos = (pred == j) & (gt == j);
            accuracy(j) = sum(true_pos)/sum(pos);
        end
    end

%% cls_confusion: function description
    function confusion_matrix = cls_confusion(pred, gt)
        num_classes = length(dataset.classes);
        confusion_matrix = zeros(num_classes, 'single');
        for j = 1:num_classes
            confusion_matrix(j,:) = histcounts(pred((gt == j)), 0.5:num_classes+0.5, 'Normalization', 'Probability');
        end
    end

    function output = score_statics(scores)
        statics = {'max','min','std','mean'};
        num_classes = length(dataset.classes);
        output = repmat(struct('max',nan,'min',nan,'std',nan,'mean',nan),num_classes,1);
        for j = 1:num_classes
            v_idx = clsName_2_testIdx(dataset.classes{j});
            cls_scores = scores(:,v_idx);
            cls_scores = cls_scores(:);
            for k = 1:length(statics)
                output(j).(statics{k}) = feval(statics{k}, cls_scores);
            end
        end
    end
end
