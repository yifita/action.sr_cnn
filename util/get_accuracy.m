function accuracy = get_accuracy(dataset, score, varargin)
ip = inputParser;
ip.addParameter('split', 1,   @isscalar);
ip.addParameter('set', 'test', @ischar);
ip.parse(varargin{:});
opts = ip.Results;
switch opts.set
    case 'test'
        gt_cls = dataset.video_cls(dataset.test_splits{opts.split});
    case 'val'
        gt_cls = dataset.video_cls(dataset.val_splits{opts.split});
    otherwise
        error('invalid set option')
end
clsName_to_id = containers.Map(dataset.classes, 1:length(dataset.classes));
gt_cls_ids = cell2mat(clsName_to_id.values(gt_cls));
gt_cls_ids = reshape(gt_cls_ids, 1, []);

[~, pred_cls_id] = max(score);
accuracy = nan(length(dataset.classes),1);
for j = 1:length(dataset.classes)
    pos = (gt_cls_ids == j);
    true_pos = (pred_cls_id == j) & (gt_cls_ids == j);
    accuracy(j) = sum(true_pos)/sum(pos);
end
end
