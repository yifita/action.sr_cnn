function model = VGG16_for_Faster_RCNN_VOCplusILSVRC(model)
% VGG 16layers (only finetuned from conv3_1)

model.mean_image                                = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_proposal_net_file             = fullfile(pwd, 'output', 'faster_rcnn_final', 'voc0712_ilsvrc_default', 'proposal_final');
model.pre_trained_detection_net_file            = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'vgg_16layers_conv3_1', 'solver_60k80k.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_proposal_net_file;
model.stage1_rpn.snapshot                       = '';

% rpn test setting
% only save proposal with confidence larger thres
model.stage1_rpn.rpn_keep_thres                 = 0.01;
model.stage1_rpn.nms.per_nms_topN              	= -1;
model.stage1_rpn.nms.nms_overlap_thres      	= 0.7;
model.stage1_rpn.nms.after_nms_topN         	= 2000;

%% stage 1 fast rcnn, inited from pre-trained network
model.stage1_fast_rcnn.solver_def_file          = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'solver_60k150k.prototxt');
model.stage1_fast_rcnn.test_net_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');
model.stage1_fast_rcnn.init_net_file            = model.pre_trained_detection_net_file;
model.stage1_fast_rcnn.snapshot                 = '';

%% stage 2 rpn, only finetune fc layers
model.stage2_rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'vgg_16layers_fc6', 'solver_60k80k.prototxt');
model.stage2_rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'vgg_16layers_fc6', 'test.prototxt');
model.stage2_rpn.snapshot                       = '';

% rpn test setting
% only save proposal with confidence larger thres
model.stage2_rpn.rpn_keep_thres                 = 0.01;
model.stage2_rpn.nms.per_nms_topN              	= -1;
model.stage2_rpn.nms.nms_overlap_thres         	= 0.7;
model.stage2_rpn.nms.after_nms_topN           	= 2000;

%% stage 2 fast rcnn, only finetune fc layers
model.stage2_fast_rcnn.solver_def_file          = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_fc6', 'solver_40k100k.prototxt');
model.stage2_fast_rcnn.test_net_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_fc6', 'test.prototxt');
model.stage2_fast_rcnn.snapshot                 = '';

%% final test
model.final_test.nms.per_nms_topN              	= 6000; % to speed up nms
model.final_test.nms.nms_overlap_thres         	= 0.7;
model.final_test.nms.after_nms_topN           	= 300;
end
