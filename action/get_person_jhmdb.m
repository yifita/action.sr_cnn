function [data, invalid] = get_person_jhmdb(video_dataset, split, set, opts)
% GET_PERSON_JHMDB extract person cues for jhmdb dataset
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
    sprintf('%s_2_per_%.1f_thres_%d_interv',...
    video_dataset.name, opts.score_thres, opts.frame_interval),...
    '.','_');

switch set
    case 'train'
        video_inds = find(video_dataset.train_splits{split});
    case 'test'
        video_inds = find(video_dataset.test_splits{split});
    case 'trainval'
        video_inds = 1:(video_dataset.num_video);
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
            struct('person_rois',zeros(1,4, 'single')), ...
            video_dataset.num_video, 1);
    invalid = false(length(video_dataset.video_ids),1);
    for i = 1:length(video_dataset.video_ids)
        try
            tic_toc_print('%s: %s (%d/%d) \n', mfilename, video_dataset.name, i, video_dataset.num_video)
            %% extract smooth person track
            load(video_dataset.feat_paths(video_dataset.faster_rcnn_dir, i));
            % jhmdb human_bb detected from png, contains more frames than video
            % jhmdb detection only saved person bb, wrap it in a cell to use existing func
            human_bb = cellfun(@(x) {x}, human_bb(1:video_dataset.num_frames(i)), 'uni', false);
            person_id = 1;

            % Dynamic programming search from existing human detections
            flow_x_paths = video_dataset.flow_x_paths(i);
            flow_y_paths = video_dataset.flow_y_paths(i);
            data(i).person_rois = find_top_n_people(human_bb, flow_x_paths, flow_y_paths,...
                person_id, opts.frame_interval, opts.top_n_person, opts.score_thres, video_dataset.frame_size(i,2), video_dataset.frame_size(i,1));
        catch err
            warning('%s(::%d) : %s', err.stack(1).file, err.stack(1).line, err.message);
            invalid(i) =true;
        end
        assert(length(data(i).person_rois) == video_dataset.num_frames(i));
    end
    fprintf('Saving rois data to %s...', fullfile(cache_dir, cache_file));
    save(fullfile(cache_dir, cache_file), 'data');
    fprintf('done.\n');
    invalid = find(invalid);
end
data = data(video_inds);
end
