function dataset = get_dataset_jhmdb(root_dir, varargin)
% GET_DATASET_JHMDB: builds a dataset from jhmdb dataset.
% Inspired by the code imdb_from_voc.m in Fast-RCNN by Ross Girshick
% 
% dataset contains the following fields
% - name: dataset name
% - root: root diretory storing the frame images (root -> video_id -> frames.jpg)
% - cache_file: path to save the output mat structure
% - split_dir: diretory storing the split files from the official dataset web
% - faster_rcnn_dir: directory storing the extracted bounding boxes
% - mask_dir: root directory storing the ground truth puppet mask
% - human_mask(vid_id): path to the matlab file storing the ground truth puppet mask
% - classes: cell array of class names
% - num_video: 
% - video_ids: unique video names
% - video_cls: video class
% - num_frames: number of frames in the video
% - video_paths(vid_id): inline function that returns the path to a given video
% - flow_x_paths(vid_id): inline function that returns the paths to dense flow images
%   generated using TVL1 algorithm
% - flow_y_paths(vid_id): inline function that returns the paths to dense flow images
%   generated using TVL1 algorithm
% - frames_of(vid_id): inline function that returns the path to frame jpgs of a given
%   video
% - feat_paths(feat_root, vid_id): returns the path to a stored feature of this video
%   e.g. dataset.feat_paths(dataset.faster_cnn_dir, 1) gives the path of the extracted
%   bounding boxes of the first video
% - train_splits: boolean vector storing the training subset of a split
% - test_splits: boolean vector storing the test subset of a split
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

cache_file = 'imdb/cache/jhmdb_dataset.mat';
if exist(cache_file, 'file');
    ld = load(cache_file);
    dataset = ld.dataset;
    return;
end

ip = inputParser;
ip.addRequired('root_dir', @ischar);
ip.addParameter('split_dir','~/Yifan/video_data/jhmdb/splits',@ischar);
ip.addParameter('features_dir','~/Yifan/feature_data',@ischar);
ip.addParameter('mask_dir', '~/Yifan/video_data/jhmdb/puppet_mask', @ischar);
ip.parse(root_dir, varargin{:});
opts = ip.Results;

dataset.name = 'jhmdb';
dataset.root = root_dir;
dataset.cache_file = cache_file;
dataset.mask_dir = opts.mask_dir;
dataset.split_dir = opts.split_dir;
dataset.faster_rcnn_dir = '/disks/sdc/01/Yifan_sdc/feature_data/faster_rcnn_extended';

% classes
tmp = dir([dataset.split_dir '/*split1.txt']);
dataset.classes = strrep({tmp(:).name}, '_test_split1.txt','');
assert(length(dataset.classes) == 21, sprintf('Found %d (~=21) classes', length(dataset.classes)));

dataset.video_ids = cell(3000, 1);
dataset.video_cls = cell(3000, 1);

% get all videos
counter = 0;
for i = 1:length(dataset.classes)
    fid = fopen(sprintf('%s/%s_test_split1.txt', dataset.split_dir, dataset.classes{i}),'r');
    tmp = textscan(fid, '%s %*[^\n]');
    videos = tmp{1};
    num_videos = length(videos);
    dataset.video_ids(counter+1:counter+num_videos) = strrep(videos, '.avi', '');
    dataset.video_cls(counter+1:counter+num_videos) = dataset.classes(i);
    counter = counter + num_videos;
end
dataset.num_video = counter;
dataset.video_ids = dataset.video_ids(1:counter);
dataset.video_cls = dataset.video_cls(1:counter);

% splits
dataset.train_splits = cell(3,1);
dataset.test_splits = cell(3,1);
[dataset.train_splits{:}] = deal(false(dataset.num_video, 1));

for split = 1:3
    for j = 1:length(dataset.classes)
        fid = fopen(sprintf('%s/%s_test_split%d.txt', dataset.split_dir, dataset.classes{j}, split), 'r');
        tmp = textscan(fid, '%s %d');
        for i = 1:length(tmp{1})
            dataset.train_splits{split}(strcmp(tmp{1}{i}(1:end-4), dataset.video_ids)) = tmp{2}(i) == 1;
        end
    end
    dataset.test_splits{split} = ~dataset.train_splits{split};
end

dataset.video_paths = @(i) sprintf('%s/%s/%s', ...
    dataset.root, dataset.video_cls{i}, dataset.video_ids{i});
dataset.feat_paths = @(feat_root, i) sprintf('%s/%s/%s/%s',feat_root, dataset.name, dataset.video_cls{i}, [dataset.video_ids{i} '.mat']);

dataset.flow_dir = '/disks/sda/01/Yifan_sda/flow_data';
dataset.flow_x_paths = @(i) arrayfun(@(frame) sprintf('%s/%s/%s/%s/flow_x_%04d.jpg', dataset.flow_dir, dataset.name, ...
    dataset.video_cls{i}, dataset.video_ids{i}, frame), 1:dataset.num_frames(i)-1, 'uni', false);
dataset.flow_y_paths = @(i) arrayfun(@(frame) sprintf('%s/%s/%s/%s/flow_y_%04d.jpg', dataset.flow_dir, dataset.name, ...
    dataset.video_cls{i}, dataset.video_ids{i}, frame), 1:dataset.num_frames(i)-1, 'uni', false);

% puppet
dataset.human_mask = @(i) sprintf('%s/%s/%s/puppet_mask.mat', dataset.mask_dir, dataset.video_cls{i}, dataset.video_ids{i});

dataset.num_frames = nan(dataset.num_video, 1);
dataset.frame_size = nan(dataset.num_video, 2);
for i = 1:dataset.num_video
    % infer number of frames from gt bounding boxes
    puppet = matfile(dataset.human_mask(i), 'Writable', false);
    dataset.num_frames(i) = size(puppet.part_mask, 3);
    % get original resolution
    tmp = imfinfo(sprintf('%s/%s/%s/00001.png',dataset.root, dataset.video_cls{i}, dataset.video_ids{i}));
    dataset.frame_size(i, :) = [tmp.Height tmp.Width];
end

dataset.frames_of = @(i) arrayfun(@(frame) ...
    sprintf('%s/%s/%s/%05d.png',dataset.root, dataset.video_cls{i}, dataset.video_ids{i}, frame), ...
    1:dataset.num_frames(i), 'UniformOutput', false);

save(cache_file,'dataset');

end
