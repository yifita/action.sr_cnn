function train_voc_ilsvrc(varargin)
% script_faster_rcnn_VGG16()
% Faster rcnn training and testing with VGG16 model
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
% save memory use global variables
global dataset
clear fast_rcnn_generate_random_minibatch
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run('startup');

%%  input parser
ip = inputParser;
ip.addParameter('caffe_version',        'caffe', @ischar);
ip.addParameter('cache_root_dir',       '.',    @ischar);
ip.addParameter('imdb_cache_dir',       '.',    @ischar);
ip.addParameter('image_data_dir',       '/disks/sda/01/Yifan_sda/image_data',   @ischar);
ip.addParameter('gpu_id',               0,              @isscalar);
ip.addParameter('do_val',               true,           @islogical);
ip.addParameter('proposal1_snapshot',    '',           @ischar);
ip.addParameter('detection1_snapshot',   '',           @ischar);
ip.addParameter('proposal2_snapshot',    '',           @ischar);
ip.addParameter('detection2_snapshot',   '',           @ischar);
ip.parse(varargin{:});
opts = ip.Results;

%% -------------------- CONFIG --------------------

% model
model = [];
model                       = Model.VGG16_for_Faster_RCNN_VOCplusILSVRC(model);
model.stage1_rpn.snapshot = opts.proposal1_snapshot;
model.stage1_fast_rcnn.snapshot = opts.detection1_snapshot;
model.stage2_rpn.snapshot = opts.proposal2_snapshot;
model.stage2_fast_rcnn.snapshot = opts.detection2_snapshot;

% cache base
% cache_base_proposal         = 'voc0712_ilsvrc_default';
cache_base_proposal         = 'vgg16_voc0712_ilsvrc_default';
cache_base_fast_rcnn        = '';
% train/test data
dataset                     = [];
use_flipped                 = true; %TODO
Dataset.get_dataset_VOCplusILSVRC('train', use_flipped, 'imdb_cache_dir', opts.imdb_cache_dir, ...
    'dataroot', opts.image_data_dir);
if opts.do_val
    Dataset.get_dataset_VOCplusILSVRC('val', false, 'imdb_cache_dir', opts.imdb_cache_dir, ...
        'dataroot', opts.image_data_dir);
end

%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config(...
    'image_means', model.mean_image, ...
    'feat_stride', model.feat_stride,...
    'cache_root_dir' , opts.cache_root_dir);
a = datevec(now);
conf_fast_rcnn              = fast_rcnn_config(...
    'image_means', model.mean_image, ...
    'use_flipped', use_flipped, ...
    'cache_root_dir' , opts.cache_root_dir,...
    'rng_seed', a(6));
clear a;
rpn_conf_min_size();
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);

% caffe environment
caffe_mex(opts.gpu_id, opts.caffe_version);
% generate anchors and pre-calculate output size of rpn network
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
%% stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, model.stage1_rpn, opts.do_val);
% test
Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, 'train');
if opts.do_val
   Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, 'test');
end

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, model.stage1_fast_rcnn, opts.do_val);
% test
% opts.mAP                    = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% reload dataset to clear memory
if conf_proposal.target_only_gt
    rm_roidb_from_proposal()
end
% train
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, model.stage2_rpn, opts.do_val);
% test
Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, 'train');
if opts.do_val
    Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, 'test');
end

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, model.stage2_fast_rcnn, opts.do_val);

%% final test
fprintf('\n***************\nfinal test\n***************\n');

model.stage2_rpn.nms        = model.final_test.nms;
Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, 'test');
% opts.final_mAP              = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);

    %% rpn_conf_min_size: function description
    function rpn_conf_min_size()
        min_size_train = min(cellfun(@(x) min(x.sizes(:)), dataset.imdb_train, 'UniformOutput', true));
        min_size_test = [];
        if isfield('imdb_test', dataset)
            min_size_test = min(cellfun(@(x) min(x.sizes(:)), dataset.imdb_test, 'UniformOutput', true));
        end
        conf_proposal.min_size = min([min_size_test(:); min_size_train(:)]);
    end
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[3:5]);
end
