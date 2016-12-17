function model = VGG16_for_Flow_CNN(model, opts)
%VGG16_FOR_FLOW_CNN flow stream choose prototxt and set up temporary cache dir
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
sub_folder = 'flow';
suffix = '';
if opts.use_scene
    suffix = [suffix '_scene'];
end
if opts.pooled_size < 7
    suffix = [suffix '_s'];
end
if opts.use_scene && opts.top_n_person > 0
    suffix = [suffix '_' opts.merge];
end
model.mean_image                       = fullfile(...
    pwd, 'models', 'pre_trained_models', sprintf('vgg16_flow', opts.dataset), 'flow_mean.mat');

% Stride in input image pixels at the last conv layer
model.feat_stride                      = 16;

model.solver_def_file                  = fullfile(...
    pwd, 'models', 'srcnn_prototxt', sub_folder, ...
    sprintf('solver_%s_%d_p_%d_c%s.prototxt', opts.dataset, opts.top_n_person, opts.obj_per_img, suffix));
model.net_def_file                = fullfile(...
    pwd, 'models', 'srcnn_prototxt', sub_folder, ...
    sprintf('trainval_%s_%d_p_%d_c%s.prototxt', opts.dataset, opts.top_n_person, opts.obj_per_img, suffix));

model.cache_name = sprintf('conv_flow_%s_%d_p_%d_c%s', ...
    opts.dataset, opts.top_n_person, opts.obj_per_img, suffix);

model.conf.input_size = [224 224];
model.conf.merge = opts.merge;
model.conf.min_bb_length = opts.pooled_size*16;
model.conf.n_person = opts.top_n_person;
model.conf.obj_per_img = opts.obj_per_img;
model.conf.use_scene = opts.use_scene;
model.conf.PERSON_BLOB_IDX = -1;
if model.conf.n_person > 0
    model.conf.PERSON_BLOB_IDX = 0;
end
model.conf.SCENE_BLOB_IDX = model.conf.PERSON_BLOB_IDX + 1;
end
