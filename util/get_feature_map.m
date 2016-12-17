function get_feature_map(video_dataset, rois_data, net_def, modelfile, clsName, varargin)
ip = inputParser;
ip.addRequired('video_dataset',         @isstruct);
ip.addRequired('rois_data',             @isstruct);
ip.addRequired('net_def',               @ischar);
ip.addRequired('modelfile',             @ischar);
ip.addRequired('clsName',               @iscell);
ip.addParameter('caffe_version',        'caffe',        @ischar);
ip.addParameter('size_per_crop',        [480 640],      @isvector);
ip.addParameter('split',                1,              @isscalar);
ip.addParameter('gpu_id',               0,              @isscalar);
ip.addParameter('sample_per_cls',       5,              @isscalar);
ip.addParameter('rng_seed',             6,              @isscalar);
ip.addParameter('sample_per_vid',       1,              @isscalar);
ip.addParameter('baseline',             true,           @islogical);
ip.parse(video_dataset, rois_data, net_def, modelfile, clsName, varargin{:});
opts = ip.Results;

clsName_2_sample = containers.Map;
for i = 1:length(video_dataset.classes)
    clsName_2_sample(video_dataset.classes{i}) = ...
    find(strcmp(video_dataset.video_cls, video_dataset.classes{i}) & video_dataset.test_splits{opts.split});
end
parsed = regexp(net_def,'.*(flow).*(\d)_p_(x|0)_c(_scene)?_?([^/]*)','tokens');
assert(~isempty(parsed) && ~isempty(parsed{1}));
assert(~isempty(parsed) && ~isempty(parsed{1}));
model.conf.n_person     = str2double(parsed{1}{2});
model.conf.obj_per_img  = 0;
model.conf.use_scene    = ~isempty(parsed{1}{4});
model.conf.merge        = parsed{1}{end};
is_flow = ~isempty(parsed{1}{1});
if is_flow
    model.conf.input_size   = [224 224];
    opts.size_per_crop      = [256 340];
    mean_image = fullfile(pwd, 'models', 'pre_trained_models', 'vgg16_flow_ucf', 'flow_mean');
    model.conf.target_crops = flow_target_crops(opts.size_per_crop, model.conf.input_size);
else
    model.conf.input_size   = [420 560];
    mean_image = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image');
    model.conf.target_crops = target_crops(opts.size_per_crop);
end
model.conf.min_bb_length = 7*16;
s = load(mean_image);
s_fieldnames = fieldnames(s);
assert(length(s_fieldnames) == 1);
model.conf.image_means = s.(s_fieldnames{1});
clear s
disp(model.conf);

num_samples = length(clsName)*opts.sample_per_cls*opts.sample_per_vid;
counter = 0;
net_inputs = cell(num_samples, 1);
image_roidb = cell(num_samples, 1);
for i = 1:length(clsName)
    test_ind = clsName_2_sample(clsName{i});
    test_ind = randsample(test_ind, opts.sample_per_cls);
    for j = 1:length(test_ind)
        if is_flow
            [net_inputs{counter+1}, ~] = rstar_cnn_get_test_flowbatch(video_dataset, ...
                rois_data(test_ind(j)), test_ind(j), model.conf, 'crop', true, 'flip', false, ...
                'size_per_crop', opts.size_per_crop, 'batchsize', 10, 'label', false,...
                'test_seg_per_video', opts.sample_per_vid);
        else
            [image_roidb{counter+1}, valid] = prepare_rois_context_minibatch(...
                video_dataset, rois_data(test_ind(j)), test_ind(j), opts.sample_per_vid, ...
                'flip', false, 'rng_sample', false);
            if ~valid
                fprintf('Warning: discarded data due to in sufficient frames\n');
            end
            % get blobs from processed rois
            net_inputs{counter+1} = rstar_cnn_get_test_minibatch(image_roidb{counter+1}, model.conf,...
                'batchsize', 10, 'crop', true, 'flip', false, 'size_per_crop', opts.size_per_crop);
        end
        counter = counter + 1;
    end
end

%% ----------- test ------------ %%
clc;
clear mex;
clear is_valid_handle; % to clear init_key

caffe_mex(opts.gpu_id, opts.caffe_version);
fprintf('Using caffe version: %s\n',opts.caffe_version);

%%    testing
caffe_net = caffe.Net(net_def, 'test');
caffe_net.copy_from(modelfile);

% set random seed
prev_rng = seed_rand(opts.rng_seed);
caffe.set_random_seed(opts.rng_seed);

% set gpu/cpu
caffe.set_mode_gpu();

feat = cell(size(net_inputs));
for i = 1:counter
    assert(size(net_inputs{i},1) == 1);
    caffe_net.reshape_as_input(net_inputs{i});
    caffe_net.set_input_data(net_inputs{i})
    caffe_net.forward_prefilled();
    tmp = caffe_net.blobs('conv5_3').get_data();
    feat{i} = permute(tmp, [4 3 2 1]);
end
caffe.reset_all();
save('conv5_3_feat', 'feat', 'net_inputs', 'image_roidb');
%% ----------- baseline ------------ %%
if ~opts.baseline
    return;
end
if is_flow
    bs_net_def = 'models/srcnn_prototxt/flow_vgg_conv/test_ucf_0_p_0_c_scene.prototxt';
    bs_net_file = sprintf('%s/split_%d/final',...
        '/home/jsong/Yifan/Yifan-masterThesis/object2action/output/srcnn_cachedir/conv_flow_ucf_0_p_0_c_scene',...
        opts.split);
else
    bs_net_def = 'models/srcnn_prototxt/vgg_conv/test_ucf_0_p_0_c_scene.prototxt';
    bs_net_file = sprintf('%s/split_%d/final',...
        '/home/jsong/Yifan/Yifan-masterThesis/object2action/output/srcnn_cachedir/conv_ucf_0_p_0_c_scene',...
        opts.split);
end

caffe_net = caffe.Net(bs_net_def,'test');
caffe_net.copy_from(bs_net_file);
feat_bs = cell(size(net_inputs));
for i = 1:counter
    if is_flow
        caffe_net.reshape_as_input(net_inputs{i}(1));
        caffe_net.set_input_data(net_inputs{i}(1));
    else
        caffe_net.reshape_as_input(net_inputs{i}([1,3]));
        caffe_net.set_input_data(net_inputs{i}([1,3]));
    end
    caffe_net.forward_prefilled();
    tmp = caffe_net.blobs('conv5_3').get_data();
    feat_bs{i} = permute(tmp, [4 3 2 1]);
end
save('conv5_3_feat', 'feat_bs', '-append');
caffe.reset_all();
rng(prev_rng);

function crop_boxes = target_crops(size_per_crop)
% 5 crop positions * #(optional crop width) * #(optional crop height)
img_width = size_per_crop(2); img_height = size_per_crop(1);
crop_sizes = round(size_per_crop * 0.875);

crop_boxes = zeros(1, 4);
crop_sizes = crop_sizes(:, [2 1]);
% center
crop_boxes(1:2) = round(([img_width img_height] - crop_sizes)/2);
crop_boxes(3:4) = crop_boxes(1:2) + crop_sizes -1;
crop_boxes = num2cell(crop_boxes,2);
end

%% flow_target_crops: function description
function crop_boxes = flow_target_crops(size_per_crop, crop_sizes)
img_width = size_per_crop(2); img_height = size_per_crop(1);
crop_boxes(1:2) = round(([img_width img_height] - crop_sizes)/2);
crop_boxes(3:4) = crop_boxes(1:2) + crop_sizes -1;
crop_boxes = num2cell(crop_boxes,2);
end
end
