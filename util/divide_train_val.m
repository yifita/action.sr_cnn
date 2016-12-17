function dataset = divide_train_val(dataset, num_group)
if ~exist('num_group','var')
    num_group = 2;
end
cls_2_id = containers.Map(dataset.classes, 1:length(dataset.classes));
dataset.trainval_splits = cell(dataset.trainval_splits);
dataset.val_splits = cell(dataset.trainval_splits);
for split = 1:3
    dataset.trainval_splits{split} = ~ dataset.test_splits{split};
    dataset.val_splits{split} = false(size(dataset.test_splits{split}));
    trainval_idx = find(dataset.trainval_splits{split});
    gt_id = cell2mat(cls_2_id.values(dataset.video_cls(dataset.trainval_splits{split})));
    for i = 1:length(dataset.classes)
        trainval_idx_i = trainval_idx(gt_id == i);
        group_idx = regexpi(dataset.video_ids(trainval_idx_i), sprintf('v_%s_g(\\d+)_c\\d+',dataset.classes{i}),'tokens');
        group_idx = cellfun(@(x) str2double(x{1}),group_idx);
        val_g = randsample(unique(group_idx),num_group);
        val_i = false(size(trainval_idx_i));
        for j = 1:length(val_g)
            val_i(group_idx == val_g(j)) = true;
        end
        dataset.val_splits{split}(trainval_idx_i( val_i)) = true;
    end
    dataset.train_splits{split} = dataset.trainval_splits{split} & ~dataset.val_splits{split};
end
end