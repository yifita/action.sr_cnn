function save_model_path = fast_rcnn_train(conf, varargin)
% save_model_path = fast_rcnn_train(conf, dataset.imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
global dataset
%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addParamValue('do_val',          false,          @isscalar);
    ip.addParamValue('val_iters',       300,            @isscalar);
    ip.addParamValue('val_interval',    2000,           @isscalar);
    ip.addParamValue('snapshot_interval',...
                                        10000,          @isscalar);
    ip.addParamValue('solver_def_file', fullfile(pwd, 'models', 'Zeiler_conv5', 'solver.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',        fullfile(pwd, 'models', 'Zeiler_conv5', 'Zeiler_conv5'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',      'Zeiler_conv5', ...
                                                        @isstr);
    ip.addParamValue('snapshot',        '',             @ischar);

    ip.parse(conf, varargin{:});
    opts = ip.Results;

%% try to find trained model
    imdbs_name = cellfun(@(x) x.name, dataset.imdb_train, 'UniformOutput', false);
    imdbs_name = strcat(imdbs_name{:});
    cache_dir = fullfile(conf.cache_root_dir, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdbs_name);
    save_model_path = fullfile(cache_dir, 'final');
    if exist(save_model_path, 'file')
        return;
    end

%% init
    % init caffe solver
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(opts.solver_def_file);
    if isempty(opts.snapshot)
        caffe_solver.net.copy_from(opts.net_file);
    else
        caffe_solver.net.copy_from(fullfile(cache_dir, opts.snapshot));
    end

    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);

    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);

    % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end

    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);

%% making tran/val data
    fprintf('Computing mean and std for regression...');
    [bbox_means, bbox_stds]...
                            = fast_rcnn_bbox_stat(conf, dataset.roidb_train);
    fprintf('Done.\n');
    % disp(bbox_means);
    % disp(bbox_stds);
    if opts.do_val
        fprintf('Preparing validation data...');
        % fix validation data
        shuffled_inds_val = cell(length(dataset.imdb_test),1);
        % sample validation from test dataset proportionally
        for i = 1:length(dataset.imdb_test)
            shuffled_inds_val(i) = fast_rcnn_generate_random_minibatch({[]}, dataset.imdb_test{i}, conf.ims_per_batch);
            clear fast_rcnn_generate_random_minibatch
        end
        num_val_iter = sum(cellfun(@length, shuffled_inds_val));
        shuffled_inds_val = cellfun(@(x) ...
            x(1:round(length(x)/num_val_iter*opts.val_iters)), shuffled_inds_val, 'uni', false);
        [image_roidb_val]...
                                = fast_rcnn_prepare_image_roidb(conf, dataset.imdb_test, dataset.roidb_test, ...
                                    'bbox_means', bbox_means, 'bbox_stds', bbox_stds, 'sub_ind', shuffled_inds_val);
        num_val_iter = sum(cellfun(@length, shuffled_inds_val));
        assert(length(image_roidb_val) == conf.ims_per_batch * num_val_iter);
        fprintf('Done.\n');

    end

%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough
    num_classes = size(dataset.roidb_train{1}.rois(1).overlap, 2);
    check_gpu_memory(conf, caffe_solver, num_classes, opts.do_val);

%% training
    shuffled_inds = cell(length(dataset.imdb_train),1);
    train_results = [];
    val_results = [];
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();

    while (iter_ < max_iter)
        caffe_solver.net.set_phase('train');

        % generate minibatch training data
        [shuffled_inds, sub_db_inds, db_ind] = fast_rcnn_generate_random_minibatch(shuffled_inds, dataset.imdb_train, conf.ims_per_batch);
        mini_image_roidb = Faster_RCNN_Train.do_prepare_minibatch(conf,...
            dataset.imdb_train{db_ind}, dataset.roidb_train{db_ind},sub_db_inds, bbox_means, bbox_stds);
        [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
            fast_rcnn_get_minibatch(conf, mini_image_roidb);

        net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);

        rst = caffe_solver.net.get_output();
        train_results = parse_rst(train_results, rst);

        % do valdiation per val_interval iterations
        if ~mod(iter_, opts.val_interval)
            %%%%%%%%%% debug
            fprintf('\n------------------------- Iteration %d -------------------------\n', iter_);
            % fprintf('sample train_imdb: %d, image_ind: %d %d\n', db_ind, sub_db_inds);
            % cls_blob = caffe_solver.net.blobs('cls_score').get_data();
            % cls_blob = permute(cls_blob, [2 1]);
            % labels_blob = reshape(labels_blob, [], 1);
            % for i = 1:size(cls_blob, 1)
            %     fprintf('%d: labels (gt): %d (pred: %f), ', i, labels_blob(i), cls_blob(i, labels_blob(i)+1));
            %     [max_cls, max_cls_ind] = max(cls_blob(i,:));
            %     fprintf('cls (pred): %d (%f).\n', max_cls_ind-1, max_cls);
            % end
            %%%%%%%%%%
            if opts.do_val
                caffe_solver.net.set_phase('test');
                for i = 1:length(num_val_iter)
                    [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
                        fast_rcnn_get_minibatch(conf, ...
                            image_roidb_val((i-1)*conf.ims_per_batch+1:i*conf.ims_per_batch));

                    % Reshape net's input blobs
                    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
                    caffe_solver.net.reshape_as_input(net_inputs);

                    caffe_solver.net.forward(net_inputs);

                    rst = caffe_solver.net.get_output();
                    val_results = parse_rst(val_results, rst);
                end
            end
            show_state(iter_, train_results, val_results);
            train_results = [];
            val_results = [];
            diary; diary; % flush diary
        end

        % snapshot
        if ~mod(iter_, opts.snapshot_interval)
            snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        end

        iter_ = caffe_solver.iter();
    end

    % final snapshot
    snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

    diary off;
    caffe.reset_all();
    rng(prev_rng);
end

function check_gpu_memory(conf, caffe_solver, num_classes, do_val)
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough

    % generate pseudo training data with max size
    im_blob = single(zeros(max(conf.scales), conf.max_size, 3, conf.ims_per_batch));
    rois_blob = single(repmat([0; 0; 0; max(conf.scales)-1; conf.max_size-1], 1, conf.batch_size));
    rois_blob = permute(rois_blob, [3, 4, 1, 2]);
    labels_blob = single(ones(conf.batch_size, 1));
    labels_blob = permute(labels_blob, [3, 4, 2, 1]);
    bbox_targets_blob = zeros(4 * (num_classes+1), conf.batch_size, 'single');
    bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 1, 2]));
    bbox_loss_weights_blob = bbox_targets_blob;

    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};

    % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);

    if do_val
        % use the same net with train to save memory
        caffe_solver.net.set_phase('test');
        caffe_solver.net.forward(net_inputs);
        caffe_solver.net.set_phase('train');
    end
end

function model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
    bbox_stds_flatten = reshape(bbox_stds', [], 1);
    bbox_means_flatten = reshape(bbox_means', [], 1);

    % merge bbox_means, bbox_stds into the model
    bbox_pred_layer_name = caffe_solver.net.layer_names{...
        ~cellfun(@isempty, ...
            strfind(caffe_solver.net.layer_names,'bbox_pred'), 'UniformOutput', true)};
    weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
    biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
    weights_back = weights;
    biase_back = biase;

    weights = ...
        bsxfun(@times, weights, bbox_stds_flatten'); % weights = weights * stds;
    biase = ...
        biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);

    % restore net to original state
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end

function show_state(iter, train_results, val_results)
    fprintf('Training : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
        1 - mean(train_results.accuarcy.data), ...
        mean(train_results.loss_cls.data), ...
        mean(train_results.loss_bbox.data));
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
            1 - mean(val_results.accuarcy.data), ...
            mean(val_results.loss_cls.data), ...
            mean(val_results.loss_bbox.data));
    end
    fprintf('\n------------------------- ------- -------------------------\n', iter);
end
