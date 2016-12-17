function mAP = fast_rcnn_test(conf, imdbs, roidbs, varargin)
% mAP = fast_rcnn_test(conf, imdb, roidb, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    checknum = @(x) isstruct(x) || iscell(x);
    ip.addRequired('imdbs', checknum);
    ip.addRequired('roidbs', checknum);
    ip.addParamValue('net_def_file',    '', 			@isstr);
    ip.addParamValue('net_file',        '', 			@isstr);
    ip.addParamValue('cache_name',      '', 			@isstr);
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    ip.addParamValue('cache_root_dir',  '/import/mfs/ait/Yifan/faster_rcnn',          @ischar);

    ip.parse(conf, imdbs, roidbs, varargin{:});
    opts = ip.Results;

    mAP = cellfun(@(x, y) test_func(conf, x, y, opts), imdbs, roidbs, 'UniformOutput', true,...
		'ErrorHandler', @(err, varargin) fprintf('%s\n',err.message));
end

function mAP = test_func(conf, imdb, roidb, opts)

    cache_dir = fullfile(opts.cache_root_dir, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdb.name);
%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file)

%%  set cache dir
    mkdir_if_missing(cache_dir);
    num_images = length(imdb.image_ids);
    num_classes = length(imdb.classes);
    ld = load('imagenet_meta');
    classes = imdb.classes;
    try
        aboxes = cell(num_classes, 1);
        if opts.ignore_cache
            throw('');
        end
        for i = 1:num_classes
            load(fullfile(cache_dir, [classes{i} '_boxes_' imdb.name opts.suffix]));
            aboxes{i} = boxes;
        end
    catch
%%      testing
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end

        % determine the maximum number of rois in testing
        if conf.use_gpu
            max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);
        end

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);

        %heuristic: keep an average of 40 detections per class per images prior to NMS
        max_per_set = 40 * num_images;
        % heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 100;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[cls][image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes = cell(num_classes, 1);
        box_inds = cell(num_classes, 1);
        for i = 1:num_classes
            aboxes{i} = cell(length(imdb.image_ids), 1);
            box_inds{i} = cell(length(imdb.image_ids), 1);
        end

        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));

            [boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);

            for j = 1:num_classes
                cls = imdb.class_ids(j);
                inds = find(~d.gt & scores(:, cls) > thresh(j));
                if ~isempty(inds)
                    [~, ord] = sort(scores(inds, cls), 'descend');
                    ord = ord(1:min(length(ord), max_per_image));
                    inds = inds(ord);
                    cls_boxes = boxes(inds, (1+(cls-1)*4):((cls)*4));
                    cls_scores = scores(inds, cls);
                    aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                    box_inds{j}{i} = [box_inds{j}{i}; inds];
                else
                    aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                    box_inds{j}{i} = box_inds{j}{i};
                end
            end

            fprintf(' time: %.3fs\n', toc(th));

            if mod(count, 1000) == 0
                for j = 1:num_classes
                [aboxes{j}, box_inds{j}, thresh(j)] = ...
                    keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
                end
                disp(thresh);
            end
        end

        for j = 1:num_classes
            [aboxes{j}, box_inds{j}, thresh(j)] = ...
                keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
        end
        disp(thresh);

        for i = 1:num_classes

            top_scores{i} = sort(top_scores{i}, 'descend');
            if (length(top_scores{i}) > max_per_set)
                thresh(i) = top_scores{i}(max_per_set);
            end

            % go back through and prune out detections below the found threshold
            for j = 1:length(imdb.image_ids)
                if ~isempty(aboxes{i}{j})
                    I = find(aboxes{i}{j}(:,end) < thresh(i));
                    aboxes{i}{j}(I,:) = [];
                    box_inds{i}{j}(I,:) = [];
                end
            end

            save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
            boxes = aboxes{i};
            inds = box_inds{i};
            save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        fprintf('test all images in %f seconds.\n', toc(t_start));

        caffe.reset_all();
        rng(prev_rng);
    end
    % clear non gt bbox
    for i = 1:length(roidb)
        for j = 1:length(roidb.rois)
            gt_ind = roidb.rois(j).gt;
            roidb.rois(j).gt(~gt_ind) = [];
            roidb.rois(j).boxes(~gt_ind,:) = [];
            roidb.rois(j).class(~gt_ind) = [];
            roidb.rois(j).overlap(~gt_ind,:) = [];
        end
    end
    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------

    if strfind(imdb.name, 'voc')
        for model_ind = 1:num_classes
          cls = imdb.classes{model_ind};
          res(model_ind) = imdb_eval_voc(cls, aboxes{model_ind}, imdb, roidb, opts.cache_name, opts.suffix);
        end
    elseif strfind(imdb.name, 'ilsvrc')
        % ilsvrc
        res = imdb_eval_ilsvrc(aboxes, imdb, roidb, opts.cache_name, opts.suffix);
    end

    if ~isempty(res)
        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('%s Results:\n', imdb.name);
        aps = [res(:).ap]' * 100;
        for i = 1:length(aps)
            fprintf('%s\t%.4f\n', imdb.classes{i}, aps(i));
        end
        disp(mean(aps));
        fprintf('~~~~~~~~~~~~~~~~~~~~\n');
        mAP = mean(aps);
    else
        mAP = nan;
    end

    diary off;
end

function max_rois_num = check_gpu_memory(conf, caffe_net)
%%  try to determine the maximum number of rois

    max_rois_num = 0;
    for rois_num = 300:500:2000
        % generate pseudo testing data with max size
        im_blob = single(zeros(max(conf.test_scales), conf.max_size, 3, 1));
        rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_blob = permute(rois_blob, [3, 4, 1, 2]);

        net_inputs = {im_blob, rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);

        caffe_net.forward(net_inputs);
        gpuInfo = gpuDevice();

        max_rois_num = rois_num;

        if gpuInfo.AvailableMemory < 2 * 10^9  % 2GB for safety
            break;
        end
    end

end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end
