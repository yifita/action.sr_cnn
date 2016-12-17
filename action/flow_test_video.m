function avg_accuracy = flow_test_video(video_dataset, rois_data, model, varargin)
% FLOW_TEST_VIDEO script to test flow stream's action recognition performance
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('video_dataset',         @isstruct);
ip.addRequired('rois_data');
ip.addRequired('model',                 @isstruct);
ip.addParameter('test_seg_per_video',   25,         @isvector);
ip.addParameter('split',                1,      @isscalar);
ip.addParameter('use_gpu',              true,   @islogical);
ip.addParameter('cache_root_dir',       '.',    @ischar);
ip.addParameter('rng_seed',             6,      @isscalar);
ip.addParameter('batchsize',            20,     @isscalar);
ip.addParameter('from_to',              [],     @isnumeric);
ip.parse(video_dataset, rois_data, model, varargin{:});
opts = ip.Results;

cache_dir = fullfile(fileparts(model.out_model_path));

% image_means is a file, load it
s = load(model.mean_image);
s_fieldnames = fieldnames(s);
assert(length(s_fieldnames) == 1);
model.conf.image_means = s.(s_fieldnames{1});
clear s

% init caffe nt
caffe_log_file_base = fullfile(cache_dir, 'caffe_log_test');
caffe.init_log(caffe_log_file_base);
%%    testing
caffe_net = caffe.Net(model.net_def_file, 'test');
caffe_net.copy_from(model.out_model_path);

% set random seed
prev_rng = seed_rand(opts.rng_seed);
caffe.set_random_seed(opts.rng_seed);

% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end
cls_to_id = containers.Map(video_dataset.classes, 1:length(video_dataset.classes));

val_ind = find(video_dataset.test_splits{opts.split});
num_videos = length(val_ind);
accum_cls = zeros(length(video_dataset.classes),1);
count_cls = zeros(length(video_dataset.classes),1);

cls_res     = nan(length(video_dataset.classes), num_videos);
scene_cls   = nan(length(video_dataset.classes), num_videos);
person_cls  = nan(length(video_dataset.classes), num_videos);
merged_cls  = nan(length(video_dataset.classes), num_videos);
prob        = nan(length(video_dataset.classes), num_videos);

if isempty(opts.from_to)
    opts.from_to = [1 num_videos];
end
opts.from_to(2) = min(opts.from_to(2), num_videos);
tic
for i = opts.from_to(1):opts.from_to(2)
    if toc > 30
        fprintf('%s: %d/%d\n', mfilename, i, num_videos);
        tic;
    end
    % get blobs from processed rois
    if model.conf.n_person > 0
        [net_inputs, valid] = get_test_flowbatch(video_dataset, ...
            rois_data.test(i), val_ind(i), model.conf, 'crop', true, 'flip', true, ...
            'batchsize', opts.batchsize,'label', true);
    else
        [net_inputs, valid] = get_test_flowbatch(video_dataset, ...
            [], val_ind(i), model.conf, 'crop', true, 'flip', true, ...
            'batchsize', opts.batchsize,'label', true);
    end
    new_gt_cls = cls_to_id(video_dataset.video_cls{val_ind(i)});
    if exist('gt_cls', 'var') && new_gt_cls ~= gt_cls
        fprintf('%s accuracy: %.3g\n', video_dataset.classes{gt_cls}, accum_cls(gt_cls)/count_cls(gt_cls));
    end
    gt_cls = new_gt_cls;
    if ~valid
        fprintf('Warning: discarded data due to in sufficient frames\n');
        continue;
    end
    rst                 = cell(size(net_inputs,1), 1);
    person_cls_blob     = cell(size(net_inputs,1), 1);
    scene_cls_blob      = cell(size(net_inputs,1), 1);
    merged_cls_blob     = cell(size(net_inputs,1), 1);
    for j = 1:size(net_inputs,1)
        caffe_net.reshape_as_input(net_inputs(j,:));
        output = caffe_net.forward(net_inputs(j,:));
        rst{j} = output{1};
        % for analysis also get results from scene and person cls
        if model.conf.n_person > 0
            person_cls_blob{j} = caffe_net.blobs('person_cls_score').get_data();
        end
        if model.conf.use_scene
            scene_cls_blob{j} = caffe_net.blobs('scene_cls_score').get_data();
        end
        if sum([model.conf.n_person > 0, model.conf.use_scene]) > 1
            merged_cls_blob{j} = caffe_net.blobs( sprintf('%s_cls_score', model.conf.merge) ).get_data();
        end
    end
    rst = cat(2, rst{:});
    prob(:,i) = mean(rst, 2);

    if model.conf.n_person > 0
        person_cls_blob = cat(2, person_cls_blob{:});
        person_cls(:,i) = mean(person_cls_blob, 2);
    end
    if model.conf.use_scene
        scene_cls_blob = cat(2, scene_cls_blob{:});
        scene_cls(:,i)  = mean(scene_cls_blob, 2);
    end
    if sum([model.conf.n_person > 0, model.conf.use_scene]) > 1
        merged_cls_blob = cat(2, merged_cls_blob{:});
        merged_cls(:,i) = mean(merged_cls_blob, 2);
    end

    % accumulate correctly classified video for average accuracy
    [~, max_cls_ind] = max(prob(:,i));
    cls_res(i) = max_cls_ind;
    accum_cls(gt_cls) = accum_cls(gt_cls) + (max_cls_ind == gt_cls);
    count_cls(gt_cls) = count_cls(gt_cls) + 1;
end
caffe.reset_all();
rng(prev_rng);

avg_accuracy = bsxfun(@rdivide, accum_cls, count_cls);
fprintf('mAP:\n');
for i = 1:length(video_dataset.classes)
    fprintf('%s:\t %.3g\n', video_dataset.classes{i}, avg_accuracy(i));
end
if opts.from_to(2)~=num_videos
    file_name = sprintf('cls_res_%d',opts.from_to(2));
else
    file_name = 'cls_res';
end
save(fullfile(cache_dir, file_name) ,...
    'avg_accuracy', 'prob', 'scene_cls', 'person_cls', 'merged_cls');
end
