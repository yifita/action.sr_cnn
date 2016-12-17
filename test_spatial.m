function avg_accuracy = test_spatial(varargin)
% TEST_SPATIAL test script for spatial stream
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addParameter('dataset',              'ucf',                 @ischar);
ip.addParameter('caffe_version',        'caffe',    @ischar);
ip.addParameter('split',                1,                      @isscalar);
ip.addParameter('gpu_id',               0,                      @isscalar);
ip.addParameter('use_gpu',              true,                   @islogical);
ip.addParameter('cache_root_dir',       '.',                    @ischar);
ip.addParameter('pooled_size',          7,                      @isscalar);
ip.addParameter('batchsize',            25,                     @isscalar);
% use flip frame
ip.addParameter('flip',                 true,                   @islogical);
% should be same as parameter context_per_roi in prototxt
ip.addParameter('obj_per_img',          0,                      @isscalar);
% root directory to save outputs
ip.addParameter('test_ims_per_video',   25,                     @isscalar);
ip.addParameter('top_n_person',         1,                      @isscalar);
ip.addParameter('use_scene',            true,                   @islogical);
ip.addParameter('use_gt_rois',          false,                  @islogical);
ip.addParameter('merge',                'sum',                  @ischar);
ip.addParameter('frame_interval',       10,                     @isscalar);
ip.addParameter('score_thres',          0.1,                    @isscalar);
ip.addParameter('model_path',           '',                     @ischar);
ip.addParameter('input_size',           [224 224],              @ischar);
ip.parse(varargin{:})
opts = ip.Results;


%% -------------------- CONFIG --------------------
% predictable training randomization
a = datevec(now);
opts.rng_seed               = a(6);
clear a

opts.rois_conf.flip                   = opts.flip;
opts.rois_conf.top_n_person           = opts.top_n_person;
opts.rois_conf.frame_interval         = opts.frame_interval;
opts.rois_conf.score_thres            = opts.score_thres;

%% -------------------- DATA ----------------------
% prepare data in form of structure
cache_dir = './imdb/cache';
ld = load(fullfile(cache_dir, sprintf('%s_dataset', opts.dataset)));
video_dataset = ld.dataset;
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
model = Model.VGG16_for_CNN(model, opts);

if isempty(opts.model_path)
    opts.model_path = fullfile(opts.cache_root_dir, 'output', 'srcnn_cachedir', ...
        model.cache_name, sprintf('split_%d',opts.split), 'final');
end
model.out_model_path = opts.model_path;
%% -------------------- Caffe Environment --------------------
clc;
clear mex;
clear is_valid_handle; % to clear init_key

caffe_mex(opts.gpu_id, opts.caffe_version);
fprintf('Using caffe version: %s\n',opts.caffe_version);
%% -------------------- Test -----------------------
% test on validation frames and get averaged classification result for video
fprintf('############# Testing r*cnn #############\n');
avg_accuracy = spatial_test_video(video_dataset, rois_data, model, ...
    'use_gpu', opts.use_gpu,...
    'test_ims_per_video', opts.test_ims_per_video,...
    'split', opts.split,...
    'rng_seed', opts.rng_seed,...
    'batchsize', opts.batchsize);
end
