function model = VGG16_for_CNN(model, opts)
%VGG16_for_CNN spatial stream choose prototxt and set up temporary cache dir
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
%% set prefix and suffix for file names
sub_folder = 'rgb';
suffix = '';
if opts.use_scene
    suffix = [suffix '_scene'];
end
if opts.pooled_size < 7
    suffix = [suffix '_s'];
end
if sum([opts.top_n_person > 0 opts.obj_per_img > 0 opts.use_scene]) > 1
    suffix = [suffix '_' opts.merge];
end
if opts.obj_per_img > 0
    obj_str = 'x';
else
    obj_str = '0';
end

model.mean_image                       = fullfile(pwd, 'models', 'pre_trained_models', 'vgg16_rgb', 'mean_image');
% Stride in input image pixels at the last conv layer
model.feat_stride                      = 16;

model.solver_def_file                  = fullfile(...
    pwd, 'models', 'srcnn_prototxt', sub_folder, ...
    sprintf('solver_%s_%d_p_%s_c%s.prototxt', opts.dataset, opts.top_n_person, obj_str, suffix));
model.net_def_file                = fullfile(...
    pwd, 'models', 'srcnn_prototxt', sub_folder, ...
    sprintf('trainval_%s_%d_p_%s_c%s.prototxt', ...
        opts.dataset, opts.top_n_person, obj_str, suffix));

model.cache_name = sprintf('%s_%d_p_%s_c%s', ...
    opts.dataset, opts.top_n_person, obj_str, suffix);
model.cache_name = ['conv_' model.cache_name];

%% model settings
model.conf.input_size = opts.input_size;
model.conf.min_bb_length = opts.pooled_size*16; % roi_pooling has to make sure rois/stride > pooled_size
model.conf.n_person = opts.top_n_person;
model.conf.obj_per_img = opts.obj_per_img;
model.conf.use_scene = opts.use_scene;
model.conf.merge = opts.merge;
model.conf.PERSON_BLOB_IDX = -1;
model.conf.OBJ_BLOB_IDX = -1;
model.conf.SCENE_BLOB_IDX = -1;
if opts.top_n_person > 0
    model.conf.PERSON_BLOB_IDX = 0;
end
if opts.obj_per_img > 0
    model.conf.OBJ_BLOB_IDX = opts.top_n_person > 0 ;
end
if opts.use_scene
    model.conf.SCENE_BLOB_IDX = (opts.top_n_person > 0) + (opts.obj_per_img > 0) ;
end
