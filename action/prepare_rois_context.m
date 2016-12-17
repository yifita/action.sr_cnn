function [data, invalid] = prepare_rois_context(video_dataset, split, set, opts)
%PREPARE_ROIS_CONTEXT load or extract relevant bounding boxes
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
invalid = [];
if ischar(video_dataset)
    ld = load(video_dataset);
    video_dataset = ld.dataset;
clear ld;
end

cache_dir = './imdb/cache';
% TODO: hard coded top_n_person to 2
cache_file = strrep(...
    sprintf('%s_%d_per_%.1f_thres_%d_interv',...
    video_dataset.name, max(2,opts.top_n_person), opts.score_thres, opts.frame_interval),...
    '.','_');
if isfield(opts, 'use_filtered') && opts.use_filtered
    cache_file = [cache_file '_flow'];
end

try
    switch set
        case 'train_val'
            video_inds = find(video_dataset.trainval_splits{split});
        case 'train'
            video_inds = find(video_dataset.train_splits{split});
        case 'val'
            video_inds = find(video_dataset.val_splits{split});
        case 'test'
            video_inds = find(video_dataset.test_splits{split});
    end
catch
    video_inds = 1:video_dataset.num_video;
end

try
    ld = load(fullfile(cache_dir, cache_file));
    s_fieldnames = fieldnames(ld);
    assert(length(s_fieldnames) == 1);
    data = ld.(s_fieldnames{1});
    clear ld s_fieldnames;
catch
    disp('rois_conf:');
    disp(opts);
    % get person_rois and obj_rois for a subsample of frames in specific videos
    data =  repmat(...
            struct('person_rois',zeros(1,4, 'single'), ...
            'obj_rois',zeros(1,5, 'single')), length(video_inds), 1);
    invalid = false(length(video_dataset.video_ids),1);
    for i = 1:length(video_dataset.video_ids)
        try
            image_paths = video_dataset.frames_of(i);
            info = imfinfo(image_paths{1});
            tic_toc_print('%s: %s (%d/%d) \n', mfilename, video_dataset.name, i, length(video_inds))
            % rois
            [data(i).person_rois, data(i).obj_rois] = find_valid_rois_seq(video_dataset, i,  ...
                'top_n_person', opts.top_n_person, ...
                'frame_interval', opts.frame_interval, ...
                'score_thres', opts.score_thres,...
                'img_width', info.Width,...
                'img_height', info.Height);
        catch err
            warning('%s(::%d) : %s', err.stack(1).file, err.stack(1).line, err.message);
            invalid(i) =true;
        end
    end
    fprintf('Saving rois data to %s...', fullfile(cache_dir, cache_file));
    save(fullfile(cache_dir, cache_file), 'data');
    fprintf('done.\n');
    invalid = find(invalid);
end
data = data(video_inds);
end
