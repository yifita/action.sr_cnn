function data = get_raw_rois(dataset, split, set, opts)
% GET_RAW_ROIS assemble a matlab structure containing detected rois
% for the whole dataset from image-wise Faster-RCNN output
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if ischar(dataset)
    ld = load(dataset);
    dataset = ld.dataset;
    clear ld;
end

cache_dir = './imdb/cache';

cache_file = sprintf('%s_x_per',...
    dataset.name);

try
    switch set
        case 'train_val'
            video_inds = find(dataset.trainval_splits{split});
        case 'train'
            video_inds = find(dataset.train_splits{split});
        case 'val'
            video_inds = find(dataset.val_splits{split});
        case 'test'
            video_inds = find(dataset.test_splits{split});
    end
catch
    video_inds = 1:dataset.num_video;
end
try
    ld = load(fullfile(cache_dir, cache_file));
    s_fieldnames = fieldnames(ld);
    assert(length(s_fieldnames) == 1);
    data = ld.(s_fieldnames{1});
    clear ld s_fieldnames;
catch
    if ~strcmp(dataset.name, 'jhmdb')
        data =  repmat(...
            struct('person_rois',zeros(1,4, 'single'), ...
            'obj_rois',zeros(1,5, 'single')), length(video_inds), 1);
    else
        data =  repmat(...
            struct('person_rois',zeros(1,4, 'single')), length(video_inds), 1);
    end
    if strcmp(dataset.name, 'jhmdb')
        person_cls_id = 1;
    else
        % class id for person used in the detection model
        % In our trained model, this is 68
        person_cls_id = 68;
    end
    for i = 1:dataset.num_video
        ld = load(dataset.feat_paths(dataset.faster_rcnn_dir, i));
        s = fieldnames(ld);
        assert(length(s)==1);
        boxes_frames = ld.(s{1});
        if ~strcmp(dataset.name, 'jhmdb')
            data(i).person_rois = cellfun(@(frame) frame{person_cls_id}, boxes_frames, 'uni', false);
            data(i).obj_rois = cellfun(@(frame) frame{setdiff(length(frame), person_cls_id)}, boxes_frames, 'uni', false);
        else
            data(i).person_rois = cellfun(@(x) {x}, boxes_frames(1:dataset.num_frames(i)), 'uni', false);
        end        
        clear ld
    end
end
save(fullfile(cache_dir, cache_file), 'data');
