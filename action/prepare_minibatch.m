function [image_roidb, valid] = prepare_minibatch(...
    video_dataset, rois_data, video_inds, ims_per_video, varargin)
% PREPARE_MINIBATCH sample ims_per_video frame(s) from videos 
% and prepare frame+roi for a minibatch. image_roidb will be input of
% get_minibatch
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('video_dataset',     @isstruct);
ip.addRequired('rois_data');
ip.addRequired('video_inds',        @isnumeric);
% #images to sample per video
ip.addRequired('ims_per_video',     @isscalar);
ip.addParameter('frame_idx',    [],     @iscell);
ip.addParameter('skip_start',   1,     @isscalar);
ip.addParameter('flip',         true,   @islogical);
ip.parse(video_dataset, rois_data, video_inds, ims_per_video, varargin{:});
opts = ip.Results;

valid = true;
image_roidb =  repmat(...
        struct('frame_path', '', 'class_id', 0, ...
        'size', zeros(1,2,'uint16'),...
        'person_rois',zeros(1,4, 'single'), ...
        'obj_rois',zeros(1,5, 'single')), ims_per_video*length(video_inds), 1);

counter = 0;
for v = 1:length(video_inds)
    i = video_inds(v);
    num_frame = video_dataset.num_frames(i);

    % sample frames
    if ~isempty(opts.frame_idx)
        % frame idx is given (by shuffle function)
        sampled_frames = opts.frame_idx{v};
        ims_per_video = length(sampled_frames);
    else
        ims_per_video = min(opts.ims_per_video, num_frame);
        try
            % skip the first skip_start frames if possible (tend to have worse roi detections)
            frames_per_slice = floor((num_frame-opts.skip_start+1)/ims_per_video);
            sampled_frames = opts.skip_start:frames_per_slice:num_frame;
            sampled_frames = sampled_frames(1:ims_per_video);
        catch
            frames_per_slice = floor(num_frame/ims_per_video);
            sampled_frames = 1:frames_per_slice:num_frame;
            sampled_frames = sampled_frames(1:ims_per_video);
        end
    end
    assert(length(sampled_frames) == ims_per_video);
    counter = ims_per_video+counter;

    % frame paths
    % NOTE: should be changed depending on file structure
    image_paths = video_dataset.frames_of(i);
    image_paths = image_paths(sampled_frames);
    [image_roidb((v-1)*ims_per_video+1:v*ims_per_video).frame_path] = image_paths{:};

    % class_ids
    [image_roidb((v-1)*ims_per_video+1:v*ims_per_video).class_id] = ...
        deal(find(strcmp(video_dataset.video_cls(i), video_dataset.classes)));

    % size of original images when the bounding boxes are extracted
    [image_roidb((v-1)*ims_per_video+1:v*ims_per_video).size] = deal(video_dataset.frame_size(i,:));

    % rois
    if ~isempty(rois_data)
        assert(length(rois_data(v).person_rois) == num_frame);
        % primary region (person)
        person_rois = rois_data(v).person_rois(sampled_frames);
        [image_roidb((v-1)*ims_per_video+1:v*ims_per_video).person_rois] = person_rois{:};
        % secondary region (secondary region)
        if isfield(rois_data, 'obj_rois')
            obj_rois = rois_data(v).obj_rois(sampled_frames);
            [image_roidb((v-1)*ims_per_video+1:v*ims_per_video).obj_rois] = obj_rois{:};
        end
    end

    % flip frame image and rois
    if opts.flip
        % flip with 50% probability
        flip_frames = randperm(length(sampled_frames), floor(length(sampled_frames)/2));
        for f = 1:length(flip_frames)
            j = flip_frames(f);
            if ~exist(append_flip(image_paths{j}), 'file')
                im = imread(image_paths{j});
                imwrite(fliplr(im), append_flip(image_paths{j}));
            end
            if ~isempty(rois_data)
                image_roidb((v-1)*ims_per_video+j).person_rois(:, [1, 3]) = info.Width + 1 - person_rois{j}(:, [3, 1]);
                image_roidb((v-1)*ims_per_video+j).obj_rois(:, [1, 3]) = info.Width + 1 - obj_rois{j}(:, [3, 1]);
            end
            image_roidb((v-1)*ims_per_video+j).frame_path = append_flip(image_paths{j});
        end
    end
end
image_roidb = image_roidb(1:counter);
end
function flip_image_path = append_flip(image_path)
[filedir, filename, extension] = fileparts(image_path);
flip_image_path = sprintf('%s/%s_flip%s',filedir, filename, extension);
end
