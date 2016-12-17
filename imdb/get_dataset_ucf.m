function dataset = get_dataset_ucf(dataset, root_dir, varargin)
%% GET_DATASET_UCF: builds a dataset from ucf dataset.
% Inspired by the code imdb_from_voc.m in Fast-RCNN by Ross Girshick
% videos frames must be first converted and saved in jpg formats
%   under the structure
%   root/video_id/image_0001.jpg ...
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
% - frame_size: video (height, width)
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

ip = inputParser;
ip.addRequired('root_dir', @ischar);
ip.addParameter('split_dir','~/Yifan/video_data/ucfTrainTestlist',@ischar);
ip.addParameter('features_dir','~/Yifan/feature_data',@ischar);
ip.addParameter('cache_root_dir','.',@ischar);
ip.parse(root_dir, varargin{:});
opts = ip.Results;

cache_file = fullfile(opts.cache_root_dir, 'imdb/cache/ucf_dataset.mat');
if exist(cache_file, 'file');
    ld = load(cache_file);
    dataset = ld.dataset;
    return;
end

dataset.name = 'ucf';
dataset.root = root_dir;
dataset.cache_file = cache_file;
dataset.split_dir = opts.split_dir;
dataset.faster_rcnn_dir = '/disks/sdc/01/Yifan_sdc/feature_data/faster_rcnn';

% classes
fid = fopen(fullfile(dataset.split_dir, 'classInd.txt'), 'rb');
classes = textscan(fid, '%d %s');
classes = classes{2};
dataset.classes = classes;
assert(length(dataset.classes) == 101, 'Should contain 101 action classes');

videos = dir(fullfile(root_dir, 'v_*'));
dataset.num_video = length(videos);
dataset.video_ids = cell(dataset.num_video, 1);
dataset.video_cls = cell(dataset.num_video, 1);

video_ids = {videos(:).name}; clear videos;
for i = 1:length(video_ids)
    classname_ = textscan(video_ids{i},'v %s %*[^\n]','delimiter','_');
    cls_id = find(strcmpi(classes, classname_{1}));
    video_cls{i} = classes{cls_id};
end
dataset.video_ids = video_ids;
dataset.video_cls = video_cls;
dataset.video_paths = @(i) sprintf('%s/%s',dataset.root, dataset.video_ids{i});
dataset.feat_paths = @(feat_root, i) fullfile(feat_root, dataset.name, [dataset.video_ids{i} '.mat']);;
get_path = @(i) sprintf('%s/%s',dataset.root, dataset.video_ids{i});

disp('Finding frame images...')
num_frames = nan(dataset.num_video,1);
frame_size = nan(dataset.num_video, 2);
for i = 1:dataset.num_video
    frames = dir(fullfile(feval(get_path, i), 'image*.jpg'));
    num_frames(i) = length(cell2mat(regexp({frames(:).name}, 'image_\d{4}.jpg')));
    tmp = imfinfo(sprintf('%s/%s/image_0001.jpg', ...
        dataset.root, dataset.video_ids{i}));
    frame_size(i, :) = [tmp.Height tmp.Width];
end
dataset.num_frames = num_frames;
dataset.frame_size = frame_size
dataset.frames_of = @(i) arrayfun(@(frame) ...
    sprintf('%s/%s/image_%04d.jpg', dataset.root, dataset.video_ids{i}, frame), ...
    1:dataset.num_frames(i), 'UniformOutput', false);

train_split_file = 'trainlist%02d.txt';
test_split_file = 'testlist%02d.txt';
dataset.train_splits = cell(3,1);
dataset.test_splits = cell(3,1);
dataset.train_splits(:) = {false(dataset.num_video,1)};
dataset.test_splits(:) = {false(dataset.num_video,1)};
for i = 1:3
    train_fid = fopen(fullfile(dataset.split_dir, sprintf(train_split_file, i)), 'rb');
    train_ids = textscan(train_fid, '%*s %s %*[^\n]','delimiter', {'/',' ','.'});
    test_fid = fopen(fullfile(dataset.split_dir, sprintf(test_split_file, i)), 'rb');
    test_ids = textscan(test_fid, '%*s %s %*[^\n]','delimiter', {'/',' ','.'});
    [~, ia, ~] = intersect(dataset.video_ids, train_ids{1});
    dataset.train_splits{i}(ia) = true;
    [~, ia, ~] = intersect(dataset.video_ids, test_ids{1});
    dataset.test_splits{i}(ia) = true;
    fclose(train_fid);fclose(test_fid);
end

save(dataset.cache_file,'dataset');

    function path = feat_path(feat_root, i, ext)
        if nargin == 2
            ext = '.mat';
        end
        path = fullfile(feat_root, dataset.name, [dataset.video_ids{i} ext]);
    end
end
