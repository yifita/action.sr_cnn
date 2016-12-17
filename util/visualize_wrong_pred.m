function wrong_test_idx = visualize_wrong_pred(dataset, data, output_cache, varargin)
ip = inputParser;
ip.addParameter('split', 1,   @isscalar);
ip.addParameter('set', 'test', @ischar);
ip.addParameter('classes', {}, @iscell);
ip.parse(varargin{:});
opts = ip.Results;
split = opts.split;
wrong_test_idx = [];
try
    ld = load(output_cache);
    parsed = regexp(output_cache, sprintf('(flow)?.*%s_(\\d)_p_(x|0)_c(_scene)?_?([^/]*)',dataset.name),'tokens');
catch
    ld = load(fullfile(output_cache, sprintf('split_%d', split), 'cls_res'));
    parsed = regexp(output_cache, sprintf('(flow)?.*%s_(\\d)_p_(x|0)_c(_scene)?_?([^/]*)',dataset.name),'tokens');
end
assert(~isempty(parsed{1}), 'Cannot parse file name');
switch opts.set
    case 'test'
        clsName_2_sample = containers.Map;
        for i = 1:length(dataset.classes)
            clsName_2_sample(dataset.classes{i}) = find(strcmp(dataset.video_cls(dataset.test_splits{split})', dataset.classes{i}));
        end
        sub_idx = find(dataset.test_splits{split}');
    case 'val'
        clsName_2_sample = containers.Map;
        for i = 1:length(dataset.classes)
            clsName_2_sample(dataset.classes{i}) = find(strcmp(dataset.video_cls(dataset.val_splits{split})', dataset.classes{i}));
        end
        sub_idx = find(dataset.test_splits{split}');
    case 'trainval'
        clsName_2_sample = containers.Map;
        for i = 1:length(dataset.classes)
            clsName_2_sample(dataset.classes{i}) = find(strcmp(dataset.video_cls(dataset.trainval_splits{split})', dataset.classes{i}));
        end
        sub_idx = find(dataset.trainval_splits{split}');
    otherwise
        error('invalid set option')
end
[~, pred] = max(ld.prob);
clsName_to_id = containers.Map(dataset.classes, 1:length(dataset.classes));
wrong_test_idx = containers.Map;
if isempty(opts.classes)
    opts.classes = dataset.classes;
end
if ~strcmp(opts.set,'trainval')
    for i = 1:length(opts.classes)
        test_samples = clsName_2_sample(opts.classes{i});
        tmp = sub_idx(test_samples);
        wrong_test_idx(opts.classes{i}) = tmp(pred(test_samples) ~= clsName_to_id(opts.classes{i}));
    end
    for i = 1:length(opts.classes)
        tmp = wrong_test_idx(opts.classes{i});
        if isempty(tmp)
            continue;
        end
        for j = 1:numel(tmp)
            display_human_bb_video(dataset,tmp(j),data(tmp(j)).person_rois,'frame_skip',5);
%             display_bb_video(dataset,tmp(j),false,data(tmp(j)).obj_rois);
        end
    end
else
    for i = 1:length(opts.classes)
        test_samples = clsName_2_sample(opts.classes{i});
        tmp = sub_idx(test_samples);
        if isempty(tmp)
            continue;
        end
        for j = 1:numel(tmp)
            display_human_bb_video(dataset,tmp(j),data(tmp(j)).person_rois,'frame_skip',5);
        end
    end
end