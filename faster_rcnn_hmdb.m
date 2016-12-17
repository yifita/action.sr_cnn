function faster_rcnn_hmdb(varargin)
% FASTER_RCNN_UCF: detect bouding boxes for ucf
% Adapted from Faster R-CNN
% save detection results to a cell array [{frame1} ... {frameN}}
% framen is a cell array [{obj_class1} ... {obj_classK}]
% obj_class1 is a Mx5 array with the coordinates and detect prob
% of the detected objects, i.e. [x, y, w, h, prob]

ip = inputParser;
ip.addParameter('caffe_version',        'caffe', @ischar);
ip.addParameter('cache_root_dir',       '.',            @ischar);
ip.addParameter('save_bb',              true,           @islogical);
ip.addParameter('gpu_id',               0,              @isscalar);
ip.addParameter('per_nms_topN',         6000,           @isscalar);
ip.addParameter('nms_overlap_thres',    0.7,            @isscalar);
ip.addParameter('after_nms_topN',       300,            @isscalar);
ip.addParameter('test_scales',          600,            @isscalar);
ip.addParameter('test_nms',             0.3,            @isscalar);
ip.addParameter('thres',                0.1,            @isscalar);
ip.addParameter('from_to',              [],             @isvector);
ip.parse(varargin{:});
opts = ip.Results;

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run('startup');

%% -------------------- DATA ----------------------
dataset = get_dataset_hmdb([],'/disks/sda/01/Yifan_sda/video_data/hmdb51_org', 'cache_root_dir', opts.cache_root_dir);
if ~isfield(dataset,'faster_rcnn_dir')
    dataset.faster_rcnn_dir = '/disks/sdc/01/Yifan_sdc/feature_data/faster_rcnn';
    save(dataset.cache_file, 'dataset');
end

if ~exist(dataset.faster_rcnn_dir, 'dir')
    mkdir(dataset.faster_rcnn_dir);
end

%% -------------------- CONFIG --------------------
caffe_mex(opts.gpu_id, opts.caffe_version);
fprintf('Using caffe version: %s\n',opts.caffe_version);

%% -------------------- INIT_MODEL --------------------
proposal_detection_model = [];
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final','voc0712_ilsvrc_default');

proposal_detection_model    = load_proposal_detection_model(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if proposal_detection_model.is_share_feature
    proposal_detection_model.detection_net_def = strrep(...
        proposal_detection_model.detection_net_def, '.prototxt', '_shared.prototxt');
end

proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);

% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net)

% set gpu/cpu
caffe.set_mode_gpu();
%% test and save
boxes_db = cell(dataset.num_video, 1);
num_classes = length(proposal_detection_model.classes);
invalid = [];
if isempty(opts.from_to)
    opts.from_to = [1 dataset.num_video];
else
    opts.from_to = [max(opts.from_to(1),1), min(opts.from_to(2), dataset.num_video)];
end
for j = opts.from_to(1):opts.from_to(2)
    try
        save_path = dataset.feat_paths(dataset.faster_rcnn_dir,j);

        if exist(save_path, 'file')
            continue;
        end
        frames = dataset.frames_of(j);
        boxes_frames = cell(length(frames), 1);
        tend = 0;
        for f = 1:length(frames)
            im = imread(frames{f});
            im = gpuArray(im);

            tstart = tic;
            % test proposal
            % boxes N*4 (x, y, w, h), scores N*1 objectiveness
            [boxes, scores] = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
            % non-maximum suppression N_nms*5 (x, y, w, h, score)
            aboxes = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, true);

            % test detection
            if proposal_detection_model.is_share_feature
                % boxes N_nms*(num_classes*4), scores N_nms*numclasses
                [boxes, scores] = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                    rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
                    aboxes(:, 1:4), opts.after_nms_topN);
            else
                [boxes, scores] = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                    aboxes(:, 1:4), opts.after_nms_topN);
            end
            tend = toc(tstart) + tend;
            boxes_cell = cell(num_classes, 1);
            for i = 1:num_classes
                % non-maximum suppression for one classes among all proposals
                % each cell [x0_c, y0_c, x1_c, y1_c, prob_c]
                boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
                boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, opts.test_nms), :);

                I = boxes_cell{i}(:, 5) >= opts.thres;
                boxes_cell{i} = boxes_cell{i}(I, :);
            end
            boxes_frames{f} = boxes_cell;
        end
        tic_toc_print('hmdb test: %d/%d. Avg time %fs/frame\n', j, opts.from_to(2), tend/length(frames));
        if opts.save_bb
            mkdir_if_missing(fileparts(save_path));
            save(save_path, 'boxes_frames');
        else
            boxes_db{j} = boxes_frames;
        end
    catch err
        fprintf(2, '%s (::%d): %s %s\n', err.stack(1).file, err.stack(1).line, dataset.video_ids{j}, err.message);
        invalid(end+1) = j;
    end
end
caffe.reset_all();
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
% to speed up nms
if per_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
end
% do nms
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
end
if after_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
end
end

function proposal_detection_model = load_proposal_detection_model(model_dir)
    ld                          = load(fullfile(model_dir, 'model'));
    proposal_detection_model    = ld.proposal_detection_model;
    clear ld;
    proposal_detection_model.proposal_net_def ...
    = fullfile(model_dir, proposal_detection_model.proposal_net_def);
    proposal_detection_model.proposal_net ...
    = fullfile(model_dir, proposal_detection_model.proposal_net);
    proposal_detection_model.detection_net_def ...
    = fullfile(model_dir, proposal_detection_model.detection_net_def);
    proposal_detection_model.detection_net ...
    = fullfile(model_dir, proposal_detection_model.detection_net);
end
