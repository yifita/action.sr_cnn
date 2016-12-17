function avg_accuracy = spatial_test_video(video_dataset, rois_data, model, varargin)
% SPATIAL_TEST_VIDEO test action recognition of the spatial stream
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
ip.addParameter('batchsize',            16,     @isscalar);
ip.addParameter('split',                1,      @isscalar);
ip.addParameter('use_gpu',              true,   @islogical);
ip.addParameter('cache_root_dir',       '.',    @ischar);
ip.addParameter('test_ims_per_video',   25,     @isscalar);
ip.addParameter('rng_seed',             6,      @isscalar);
ip.addParameter('channel_weights',      [],     @ismatrix);
ip.parse(video_dataset, rois_data, model, varargin{:});
opts = ip.Results;

cache_dir = fullfile(fileparts(model.out_model_path));

% image_means is a file, load it
s = load(model.mean_image);
s_fieldnames = fieldnames(s);
assert(length(s_fieldnames) == 1);
model.conf.image_means = s.(s_fieldnames{1});
clear s

% init caffe net
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
num_classes = length(video_dataset.classes);
cls_to_id = containers.Map(video_dataset.classes, 1:num_classes);

test_ind = find(video_dataset.test_splits{opts.split});
% tmp = strfind(video_dataset.video_ids, 'v_PlayingPiano_g15');
% test_ind = find(~cellfun('isempty',tmp));
% tmp = strfind(video_dataset.video_ids(video_dataset.test_splits{opts.split}), 'v_PlayingPiano_g15');
% tmp = find(~cellfun('isempty',tmp));
% rois_data.test = rois_data.test(tmp);
num_videos = length(test_ind);

accum_cls = zeros(num_classes,1);
count_cls = zeros(num_classes,1);

cls_res     = nan(num_classes, num_videos);
prob        = nan(num_classes, num_videos);

num_channels = sum([model.conf.n_person > 0, model.conf.use_scene, model.conf.obj_per_img > 0]);
cls_scores = cell(num_channels, 1);
[cls_scores{:}] = deal(nan(num_classes, num_videos));
do_merge = isfield(model.conf,'merge') && ...
    sum([model.conf.n_person > 0, model.conf.use_scene, model.conf.obj_per_img > 0]) > 1;
if do_merge
    merged_cls = nan(num_classes, num_videos);
end
blob_names = caffe_net.blob_names;
merge_blob_name = sprintf('%s_cls_score', model.conf.merge);
if ~any(strcmp(blob_names, merge_blob_name))
    merge_blob_name = blob_names{end-1};
end
tic
for i = 1:num_videos
    if toc > 30
        fprintf('%s: %d/%d\n', mfilename, i, num_videos);
        tic;
    end
    if isstruct(rois_data)
        [image_roidb, valid] = prepare_minibatch(...
            video_dataset, rois_data.test(i), test_ind(i), opts.test_ims_per_video, ...
            'flip', false);
    else
        [image_roidb, valid] = prepare_minibatch(...
            video_dataset, [], test_ind(i), opts.test_ims_per_video, 'flip', false);
    end
    if ~valid
        fprintf('Warning: discarded data due to in sufficient frames\n');
    end
    % get blobs from processed rois
    net_inputs = get_test_minibatch(image_roidb, model.conf,...
        'batchsize', opts.batchsize, 'crop', true, 'flip', true, 'label', true);

    rst                 = cell(size(net_inputs,1), 1);
    person_cls_blob     = cell(size(net_inputs,1), 1);
    scene_cls_blob      = cell(size(net_inputs,1), 1);
    obj_cls_blob        = cell(size(net_inputs,1), 1);
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
        if model.conf.obj_per_img > 0
            obj_cls_blob{j} = caffe_net.blobs('object_cls_score').get_data();
        end
        if do_merge
            merged_cls_blob{j} = caffe_net.blobs(merge_blob_name).get_data();
        end
    end
    rst = cat(2, rst{:});
    prob(:,i)       = mean(rst, 2);

    new_gt_cls = cls_to_id(video_dataset.video_cls{test_ind(i)});
    if exist('gt_cls', 'var') && new_gt_cls ~= gt_cls
        fprintf('%s accuracy: %.3g\n', video_dataset.classes{gt_cls}, accum_cls(gt_cls)/count_cls(gt_cls));
    end
    gt_cls = new_gt_cls;
    if model.conf.n_person > 0
        person_cls_blob = cat(2, person_cls_blob{:});
        score = mean(person_cls_blob, 2);
        cls_scores{model.conf.PERSON_BLOB_IDX+1}(:,i) = score;
    end
    if model.conf.use_scene
        scene_cls_blob = cat(2, scene_cls_blob{:});
        score  = mean(scene_cls_blob, 2);
        cls_scores{model.conf.SCENE_BLOB_IDX+1}(:,i) = score;
    end
    if model.conf.obj_per_img > 0
        obj_cls_blob = cat(2, obj_cls_blob{:});
        score  = mean(obj_cls_blob, 2);
        cls_scores{model.conf.OBJ_BLOB_IDX+1}(:,i) = score;
    end
    if do_merge
        merged_cls_blob = cat(2, merged_cls_blob{:});
        score  = mean(merged_cls_blob, 2);
        merged_cls(:,i) = score;
    end

    [~, max_cls_ind] = max(prob(:,i));
    cls_res(i) = max_cls_ind;
    accum_cls(gt_cls) = accum_cls(gt_cls) + (max_cls_ind == gt_cls);
    count_cls(gt_cls) = count_cls(gt_cls) + 1;
end
caffe.reset_all();
rng(prev_rng);

avg_accuracy = bsxfun(@rdivide, accum_cls, count_cls);
fprintf('mAP:\n');
for i = 1:num_classes
    fprintf('%s:\t %.3g\n', video_dataset.classes{i}, avg_accuracy(i));
end
save(fullfile(cache_dir, 'cls_res') ,'avg_accuracy', 'prob');

if model.conf.use_scene
    scene_cls = cls_scores{model.conf.SCENE_BLOB_IDX+1};
    save(fullfile(cache_dir, 'cls_res'), 'scene_cls', '-append');
end
if model.conf.n_person > 0
    person_cls = cls_scores{model.conf.PERSON_BLOB_IDX+1};
    save(fullfile(cache_dir, 'cls_res'), 'person_cls', '-append');
end
if model.conf.obj_per_img > 0
    obj_cls = cls_scores{model.conf.OBJ_BLOB_IDX+1};
    save(fullfile(cache_dir, 'cls_res'), 'obj_cls', '-append');
end
if ~isempty(opts.channel_weights)
    weighted_cls = opts.channel_weights' * cell2mat(cls_scores);
    save(fullfile(cache_dir, 'cls_res'), 'weighted_cls', '-append');
end
if do_merge
    save(fullfile(cache_dir, 'cls_res'), 'merged_cls', '-append');
end
end
