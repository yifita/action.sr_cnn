function dataset = get_dataset_hmdb(dataset, root_dir, varargin)
% GET_DATASET_HMDB: builds a dataset from hmdb dataset.
% Inspired by the code imdb_from_voc.m in Fast-RCNN by Ross Girshick
% videos frames must be first converted and saved in jpg formats
%   under the structure
%   root/video_cls/video_id/image_0001.jpg ...
% 
% dataset contains the following fields
% - name: dataset name
% - root: root diretory storing the frame images (root -> video_id -> frames.jpg)
% - cache_file: path to save the output mat structure
% - split_dir: diretory storing the split files from the official dataset web
% - faster_rcnn_dir: directory storing the extracted bounding boxes
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

cache_file = 'imdb/cache/hmdb_dataset.mat';
if exist(cache_file, 'file');
    ld = load(cache_file);
    dataset = ld.dataset;
    return;
end

ip = inputParser;
ip.addRequired('root_dir', @ischar);
ip.addParameter('split_dir','~/Yifan/video_data/hmdbTestTrainMulti_7030_splits',@ischar);
ip.addParameter('features_dir','~/Yifan/feature_data',@ischar);
ip.parse(root_dir, varargin{:});
opts = ip.Results;

dataset.name = 'hmdb';
dataset.root = root_dir;
dataset.cache_file = cache_file;
dataset.split_dir = opts.split_dir;
dataset.faster_rcnn_dir = '/disks/sdc/01/Yifan_sdc/feature_data/faster_rcnn';

% classes
folderlist = dir(root_dir);
foldername = {folderlist(:).name};
dataset.classes = setdiff(foldername,{'.','..'});
assert(length(dataset.classes) == 51, 'Should contain 51 action classes');

dataset.video_ids = cell(7000, 1);
dataset.video_cls = cell(7000, 1);
counter = 0;

for i = 1:length(dataset.classes)
    curr_dir = fullfile(root_dir, dataset.classes{i});
    videos = dir(curr_dir);
    videos = setdiff({videos([videos(:).isdir]).name},{'.','..'});
    num_videos = length(videos);
    dataset.video_ids(counter+1:counter+num_videos) = videos;
    dataset.video_cls(counter+1:counter+num_videos) = dataset.classes(i);
    counter = counter + num_videos;
end
dataset.num_video = counter;
dataset.video_ids = dataset.video_ids(1:counter);
dataset.video_cls = dataset.video_cls(1:counter);

dataset.video_paths = @(i) sprintf('%s/%s/%s', dataset.root, dataset.video_cls{i}, dataset.video_ids{i});
dataset.feat_paths = @(feat_root, i) sprintf('%s/%s/%s/%s', feat_root, dataset.name, dataset.video_cls{i}, [dataset.video_ids{i} '.mat']);

disp('Finding frame images...')
dataset.num_frames = nan(dataset.num_video,1);
for i = 1:dataset.num_video
    frames = dir(dataset.video_paths(i));
    is_jpgs = ~cellfun(@isempty, strfind({frames(:).name}, '.jpg'), 'UniformOutput',true);
    dataset.num_frames(i) = sum(is_jpgs);
    tmp = imfinfo(sprintf('%s/%s/%s/image_0001.jpg', ...
        dataset.root, dataset.video_cls{i}, dataset.video_ids{i}));
    dataset.frame_size(i, :) = [tmp.Height tmp.Width];
end

dataset.frames_of = @(i) arrayfun(@(frame) ...
    sprintf('%s/%s/%s/image_%04d.jpg', dataset.root, dataset.video_cls{i}, dataset.video_ids{i}, frame), ...
    1:dataset.num_frames(i), 'UniformOutput', false);

dataset.flow_dir = '/disks/sda/01/Yifan_sda/flow_data';
dataset.flow_x_paths = @(i) arrayfun(@(frame) sprintf('%s/%s/%s/%s/flow_x_%04d.jpg', dataset.flow_dir, dataset.name, ...
    dataset.video_cls{i}, dataset.video_ids{i}, frame), 1:dataset.num_frames(i), 'uni', false);
dataset.flow_y_paths = @(i) arrayfun(@(frame) sprintf('%s/%s/%s/%s/flow_y_%04d.jpg', dataset.flow_dir, dataset.name, ...
    dataset.video_cls{i}, dataset.video_ids{i}, frame), 1:dataset.num_frames(i), 'uni', false);

% splits
split_file = '%s_test_split%d.txt';
dataset.train_splits = cell(3,1);
dataset.test_splits = cell(3,1);
for i = 1:3
    dataset.train_splits{i} = false(dataset.num_video, 1);
    dataset.test_splits{i} = false(dataset.num_video, 1);

    disp('Reading split text files...');
    counter = 1;
    for c = 1:numel(dataset.classes)
        fid = fopen(fullfile(dataset.split_dir,  sprintf(split_file, dataset.classes{c}, i)), 'r');
        C = textscan(fid,'%s %d');
        video_ids = strcat(dataset.video_ids(strcmp(dataset.video_cls, dataset.classes{c})), '.avi');

        assert(all(strcmp(video_ids, C{1})), 'Split file names in consistent with dataset');

        dataset.train_splits{i}(counter:counter+length(video_ids)-1) = C{2} == 1;
        dataset.test_splits{i}(counter:counter+length(video_ids)-1) = C{2} == 2;
        counter = counter + numel(video_ids);
        fclose(fid);
    end
end
save(dataset.cache_file,'dataset');
    function out = feat_path(in)
        out = [];
    end
end
function is_avi = filter_avi(directory)
[~, ~, ext ] = fileparts(directory);
is_avi = strcmp(ext, '.avi');
end
