function caffe_mex(gpu_id, caffe_version)
% caffe_mex(gpu_id, faster_rcnn_dir)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    % set gpu in matlab
    faster_rcnn_dir = '/home/jsong/Documents/faster_rcnn';
    limin_caffe_dir = '/home/jsong/Documents/caffe-limin';
    gpuDevice(gpu_id+1);

    cur_dir = pwd;

    %addpath(genpath(fullfile(faster_rcnn_dir, 'external', caffe_version, 'matlab')));
    addpath(genpath(fullfile(limin_caffe_dir, 'matlab')));
    caffe.set_device(gpu_id);
end
