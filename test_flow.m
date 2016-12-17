function avg_accuracy = test_flow(varargin)
% TEST_FLOW test script for flow stream
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addParameter('dataset',              'ucf',                  @ischar);
ip.addParameter('caffe_version',        'caffe',                @ischar);
ip.addParameter('split',                1,                      @isscalar);
ip.addParameter('gpu_id',               0,                      @isscalar);
ip.addParameter('cache_root_dir',       '.',                    @ischar);
ip.addParameter('pooled_size',          7,                      @isscalar);
ip.addParameter('merge',                'sum',                  @ischar);
ip.addParameter('batchsize',            50,                     @isscalar);
ip.addParameter('test_seg_per_video',   25,                     @isscalar);
% use flip frame
ip.addParameter('flip',                 true,                   @islogical);
% root directory to save outputs
ip.addParameter('top_n_person',         1,                      @isscalar);
% should be same as parameter context_per_roi in prototxt
ip.addParameter('obj_per_img',          0,                      @isscalar);
ip.addParameter('use_scene',            true,                   @islogical);
ip.addParameter('frame_interval',       10,                     @isscalar);
ip.addParameter('score_thres',          0.1,                    @isscalar);
ip.addParameter('model_path',           '',                     @ischar);
ip.addParameter('from_to',              [],                     @isvector);
ip.addParameter('use_gt_rois',          false,                  @islogical);
ip.parse(varargin{:})
opts = ip.Results;

%% -------------------- Caffe Environment --------------------
clc;
clear mex;
clear is_valid_handle; % to clear init_key

%% -------------------- CONFIG --------------------
caffe_mex(opts.gpu_id, opts.caffe_version);
fprintf('Using caffe version: %s\n',opts.caffe_version);

% predictable training randomization
a = datevec(now);
opts.rng_seed               = a(6);
clear a

opts.rois_conf.flip                   = opts.flip;
opts.rois_conf.top_n_person           = opts.top_n_person;
opts.rois_conf.frame_interval         = opts.frame_interval;
opts.rois_conf.score_thres            = opts.score_thres;
opts.rois_conf.use_filtered           = false;
%% -------------------- DATA ----------------------
% prepare data in form of structure
cache_dir = './imdb/cache';
ld = load(fullfile(cache_dir, sprintf('%s_dataset', opts.dataset)));
video_dataset = ld.dataset;
if ~isfield(video_dataset, 'flow_dir')
    video_dataset.flow_dir = '/disks/sda/01/Yifan_sda/flow_data';
end
fprintf('############# Loading ROIs data #############\n');
rois_data = [];
if opts.top_n_person > 0 || opts.obj_per_img > 0
    if opts.use_gt_rois
        assert(strcmp(opts.dataset, 'jhmdb'),'only jhmdb dataset has ground truth!');
        ld = load('imdb/cache/jhmdb_BB_gt');
        rois_data.test = ld.data(video_dataset.test_splits{opts.split});
        clear ld;
    else
        rois_data.test = prepare_rois_context(video_dataset, opts.split, 'test', opts.rois_conf);
    end
end

%% -------------------- MODEL ----------------------
model = [];
model = Model.VGG16_for_Flow_CNN(model, opts);

if isempty(opts.model_path)
    opts.model_path = fullfile(opts.cache_root_dir, 'output', 'srcnn_cachedir', ...
        model.cache_name, sprintf('split_%d',opts.split), 'final');
end
model.out_model_path = opts.model_path;
%% -------------------- Test -----------------------
% test on validation frames and get averaged classification result for video
fprintf('############# Testing r*cnn #############\n');
avg_accuracy = flow_test_video(video_dataset, rois_data, model, ...
    'split', opts.split,...
    'test_seg_per_video', opts.test_seg_per_video,...
    'rng_seed', opts.rng_seed,...
    'batchsize', opts.batchsize,...
    'from_to', opts.from_to);
end
