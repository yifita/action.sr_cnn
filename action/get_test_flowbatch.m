function [net_inputs, valid] = get_test_flowbatch(...
    video_dataset, rois_data, video_ind, conf, varargin)
% GET_TEST_FLOWMINIBATCH create a testing minibatch for flow stream
 % using the frame+roi generated from prepare_minibatch
 % ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('video_dataset',     @isstruct);
ip.addRequired('rois_data');
ip.addRequired('video_ind',        @isnumeric);
ip.addRequired('conf',        		@isstruct);
ip.addParameter('segment_length',       10,     @isscalar);
ip.addParameter('test_seg_per_video',   25,     @isvector);
ip.addParameter('flip',                 true,   @islogical);
ip.addParameter('crop',                 true,   @islogical);
ip.addParameter('upscale',          1.2,        @isscalar);
ip.addParameter('batchsize',        16,         @isscalar);
ip.addParameter('label',            true,       @islogical);
% threshold person bb aoi to drop a sample
ip.addParameter('min_inter_ratio',  0.5,    @isscalar);
ip.parse(video_dataset, rois_data, video_ind, conf, varargin{:});
opts = ip.Results;

%% return empty as error handler
    function output = returnEmpty(s,input)
        output = [];
    end
%% returnSceneRoi: return scene rois as error handler
    function output = returnSceneRoi(s, input)
        output = [1 1 conf.input_size(2) conf.input_size(1)];
    end
%% flip_rois: flip rois [batch_ind x0 y0 x1 y1]
    function rois = flip_rois(rois)
        rois(:, [4 2]) = conf.input_size(2)-rois(:,[2 4])+1;
    end

net_inputs = {};
valid = true;
num_frame = video_dataset.num_frames(video_ind);
if num_frame < opts.segment_length
    valid = false;
    return;
end
% frame indices in each segment
step = max(floor((num_frame-opts.segment_length+1)/opts.test_seg_per_video),1);
start_frames = 1:step:(num_frame-opts.segment_length+1);
start_frames = start_frames(1:min(opts.test_seg_per_video, length(start_frames)));
num_segments = length(start_frames);
% load flows
flow_x_paths = video_dataset.flow_x_paths(video_ind);
flow_y_paths = video_dataset.flow_y_paths(video_ind);
% flow_x_paths = arrayfun(@(frames) sprintf('%s/%s/%s/%s/flow_x_%04d.jpg',...
%      video_dataset.flow_dir, video_dataset.name, video_dataset.video_cls{video_ind}, video_dataset.video_ids{video_ind}, frames), 1:video_dataset.num_frames(video_ind), 'uni',false);
% flow_y_paths = arrayfun(@(frames) sprintf('%s/%s/%s/%s/flow_y_%04d.jpg',...
%      video_dataset.flow_dir, video_dataset.name, video_dataset.video_cls{video_ind}, video_dataset.video_ids{video_ind}, frames), 1:video_dataset.num_frames(video_ind), 'uni',false);
info = imfinfo(flow_x_paths{1});
img_size = [info.Height info.Width];
flows = zeros(img_size(1), img_size(2), opts.segment_length*2, num_segments, 'single');
for i = 1:num_segments
    for j = 1:opts.segment_length
        img_x = imread(flow_x_paths{start_frames(i)+j-1});
        img_y = imread(flow_y_paths{start_frames(i)+j-1});
        flows(:,:,(j-1)*2+1,i) = img_x;
        flows(:,:,(j-1)*2+2,i) = img_y;
    end
end
if opts.flip
    num_segments = num_segments*2;
end
im_blob = cell(num_segments, 1);
if conf.n_person > 0
    person_blob = cell(num_segments, 1);
end
if opts.label
    label_blob = cell(num_segments, 1);
    [label_blob{:}] = deal(find(strcmp(video_dataset.video_cls{video_ind}, video_dataset.classes))-1);
end
[conf.target_crops, size_per_crop] = target_crops(img_size, conf.input_size);
counter_batch = 0;
for i = 1:length(start_frames)
    %% get image_roidb
    % pick one frame randomly
    start_frame = start_frames(i);

    % choose one person_rois
    if ~isempty(rois_data) && conf.n_person>0
        assert(length(rois_data.person_rois) == num_frame);
        person_rois_tmp = rois_data.person_rois(start_frame:(start_frame+opts.segment_length-1));

        % primary region (person)
        try
            num_persons = size(person_rois_tmp{round(opts.segment_length/2)}, 1);
            person_rois = zeros(num_persons, 4);
            for k=1:num_persons
                person_tube = cellfun(@(x) x(k, 1:4), person_rois_tmp, 'uni', false, 'ErrorHandler', @returnEmpty);
                person_tube = cat(1,person_tube{:});
                if size(person_rois,1)>0
                    % find the largest bounding box
                    person_rois(k, :) = [min(person_tube(:,1)) min(person_tube(:,2)) max(person_tube(:,3)) max(person_tube(:,4))];
                    % person_rois extracted from original resolution, flow image has been changed
                    if strcmp(video_dataset.name, 'ucf') && ~isequal(img_size, [240, 320])
                        person_rois = fast_rcnn_map_im_rois_to_feat_rois([], person_rois, img_size./[240 320]);
                    end
                    % upscale person_rois
                    if opts.upscale ~= 1
                        person_center = [person_rois(k, 3)+person_rois(k, 1) person_rois(k, 4)+person_rois(k, 2)]/2;
                        person_rois(k,:) = (person_rois(k,:) - person_center(:,[1 2 1 2])) * opts.upscale + person_center(:,[1 2 1 2]);
                        person_rois(k, 1) = max(person_rois(k, 1), 1);
                        person_rois(k, 2) = max(person_rois(k, 2), 1);
                        person_rois(k, 3) = min(person_rois(k, 3), img_size(2));
                        person_rois(k, 4) = min(person_rois(k, 4), img_size(1));
                    end
                end
            end
        catch
            person_rois = [1 1 img_size(2) img_size(1)];
        end
    end

    %% assemble actual batch
    [im, im_scale] = get_image_blob(i);
    sample_in_crop = ones(length(im), 1);
    if conf.n_person > 0 && ~isempty(rois_data)
        if opts.crop
            person_rois = fast_rcnn_map_im_rois_to_feat_rois([], person_rois, size_per_crop./img_size);
            inters = cellfun(@(crop) boxinter(person_rois, crop, opts.min_inter_ratio), conf.target_crops, 'uni', false);
            % process persons with valid cropped size
            person_rois = cellfun(@(inter, crop, scale) ...
                get_cropped_rois(person_rois(inter, :), crop, scale), ...
                inters, conf.target_crops, im_scale, 'uni', false, 'ErrorHandler', @returnSceneRoi);
        else
            person_rois = cellfun(@(scale) fast_rcnn_map_im_rois_to_feat_rois(...
                [], person_rois, scale), im_scale, 'uni', false);
        end
        person_rois = cellfun(@(x) ...
            check_rois_pool(x, conf.min_bb_length, conf.input_size), person_rois, 'uni', false);
        person_rois(cellfun('isempty', person_rois)) = {[1 1 conf.input_size(2) conf.input_size(1)]};
        sample_in_crop = cellfun(@(x) size(x, 1), person_rois);

    end
    batch_ind = mat2cell([counter_batch+1:counter_batch+sum(sample_in_crop)]', sample_in_crop);
    counter_batch = counter_batch + sum(sample_in_crop);
    sample_in_crop_cell = num2cell(sample_in_crop, 2);

    % put current segment input into global inputs
    if conf.n_person > 0
        person_rois = cellfun(@(ind, rois) [ind rois], batch_ind, person_rois, 'uni', false);
        person_blob{i} = cat(1, person_rois{:});
    end
    im = cellfun(@(x, y) repmat(x, [1 1 1 y]), im, sample_in_crop_cell, 'uni', false);
    im_blob{i} = cat(4, im{:});
    if opts.label
        label = cellfun(@(y) repmat(label_blob{i}, y, 1), sample_in_crop_cell, 'uni', false);
        label_blob{i} = cat(1, label{:});
    end
end
if opts.flip
    tmp = num_segments/2;
    im_blob(tmp+1:num_segments) = ...
        cellfun(@fliplr, im_blob(1:tmp),'uni', false);
    if conf.n_person > 0
        person_blob(tmp+1:num_segments) = ...
            cellfun(@flip_rois, person_blob(1:tmp), 'uni', false);
        person_blob(tmp+1:num_segments) = ...
            cellfun(@(x) [x(:,1)+counter_batch x(:,2:end)], person_blob(tmp+1:num_segments), 'uni', false);
    end
    if opts.label
        label_blob(tmp+1:num_segments) = label_blob(1:tmp);
    end
end

%% divide to fit batchsize
% permute data into caffe c++ memory, thus [num, channels, height, width]
im_blob = cat(4, im_blob{:});
im_blob = permute(im_blob, [2, 1, 3, 4]);
num_blobs = ceil(size(im_blob,4)/opts.batchsize);
total_batches = size(im_blob,4);

im_blob = arrayfun(@(i) ...
	im_blob(:,:,:,(i-1)*opts.batchsize+1:min(total_batches, i*opts.batchsize)),1:num_blobs,'uni',false);
im_blob = reshape(im_blob, [], 1);

if conf.n_person > 0
    person_blob = cat(1, person_blob{:});
    person_blob = person_blob - 1; % to c's index (start from 0)
    person_blob = divide_rois_blob(person_blob, num_blobs, opts.batchsize);
    person_blob = cellfun(@(x) single(permute(x, [3, 4, 2, 1])), person_blob, 'uni',false);
end
if conf.use_scene
    scene_rois = repmat([1 1 conf.input_size(2) conf.input_size(1)], total_batches, 1);
    scene_blob = [reshape(1:total_batches, [], 1) scene_rois];
    scene_blob = scene_blob - 1; % to c's index (start from 0)
    scene_blob = divide_rois_blob(scene_blob, num_blobs, opts.batchsize);
    scene_blob = cellfun(@(x) single(permute(x, [3, 4, 2, 1])), scene_blob, 'uni', false);
end

if conf.n_person > 0 && ~isempty(rois_data) && ~conf.use_scene
    net_inputs = [im_blob, person_blob];
elseif conf.n_person == 0 && conf.use_scene
    net_inputs = [im_blob];
elseif conf.n_person > 0 && conf.use_scene
    net_inputs = [im_blob, person_blob, scene_blob];
end
if opts.label
    label_blob = cat(1,label_blob{:});
    label_blob = arrayfun(@(i) label_blob((i-1)*opts.batchsize+1:min(total_batches, i*opts.batchsize)),1:num_blobs,'uni',false);
    label_blob = cellfun(@(x) single(permute(x, [3 4 2 1])), label_blob, 'uni',false);
    net_inputs = horzcat(net_inputs, reshape(label_blob, [], 1));
end
    function [im, im_scale] = get_image_blob(seg_ind)
        im = flows(:,:,:,seg_ind);
        im = bsxfun(@minus, im, conf.image_means);
        if ~opts.crop
            im_scale = {conf.input_size./img_size};
            im = {imresize(im, conf.input_size, 'bilinear', 'antialiasing', false)};
        else
            % crop
            if ~isequal(size(im(:,:,1)), size_per_crop)
                im = imresize(im, size_per_crop, 'bilinear', 'antialiasing', false);
            end
            im = cellfun(@(crop) im(crop(2):crop(4), crop(1):crop(3), :), conf.target_crops, 'uni', false);
            im_scale = cellfun(@(x) conf.input_size./size(x(:,:,1)), im, 'uni', false);
            assert(all(cellfun(@(x) isequal(x, [1 1]), im_scale)));
        end
    end
end

function rois = check_rois_pool(rois, min_bb_length, input_size)
if isempty(rois)
    return
end
w = rois(:,3)-rois(:,1)+1;
h = rois(:,4)-rois(:,2)+1;
invalid_w = find(w < min_bb_length);
invalid_h = find(h < min_bb_length);
if ~isempty(invalid_w)
    rois(invalid_w, 1) = round(rois(invalid_w, 1) - (min_bb_length - w(invalid_w))/2);
    rois(invalid_w, 3) = rois(invalid_w, 1) + min_bb_length - 1;
    % handle out of border
    tmp = rois(:,1) < 1;
    rois(tmp, [1 3]) = repmat([1 min_bb_length], sum(tmp), 1);
    tmp = rois(:,3) > input_size(2);
    rois(tmp, [1 3]) = repmat([input_size(2)-min_bb_length+1 input_size(2)], sum(tmp), 1);
end
if ~isempty(invalid_h)
    rois(invalid_h, 2) = round(rois(invalid_h, 2) - (min_bb_length - h(invalid_h))/2);
    rois(invalid_h, 4) = rois(invalid_h, 2) + min_bb_length - 1;
    tmp = rois(:,2) < 1;
    rois(tmp, [2 4]) = repmat([1 min_bb_length], sum(tmp), 1);
    tmp = rois(:,4) > input_size(1);
    rois(tmp, [2 4]) = repmat([input_size(1)-min_bb_length+1 input_size(1)], sum(tmp), 1);
end
end

%% crop rois
function rois = get_cropped_rois(rois, crop_box, im_scale_to_target)
if isempty(rois)
    return
end
rois(:, 1) = max(rois(:,1), crop_box(1));
rois(:, 2) = max(rois(:,2), crop_box(2));
rois(:, 3) = min(rois(:,3), crop_box(3));
rois(:, 4) = min(rois(:,4), crop_box(4));
rois = bsxfun(@minus, rois, crop_box([1 2 1 2]))+1;
% set invalid rois to [1 1 1 1]
w = rois(:,3)-rois(:,1)+1;
h = rois(:,4)-rois(:,2)+1;
valid = w>0 & h>0;
rois(~valid,:) = [];
rois = fast_rcnn_map_im_rois_to_feat_rois([], rois, im_scale_to_target);
end

%% divide rois blob to fit batch size
function [rois_blobs, empty_blob] = divide_rois_blob(rois_blob, num_blobs, batchsize)
rois_blobs = cell(num_blobs,1);
empty_blob = [];
for i = 0:num_blobs-1
    try
        idx = (rois_blob(:,1) >= i*batchsize) & (rois_blob(:,1) < (i+1)*batchsize);
        rois_blobs{i+1} = rois_blob(idx,:);
        rois_blobs{i+1}(:,1) = rois_blobs{i+1}(:,1)-rois_blobs{i+1}(1,1);
    catch
        rois_blobs{i+1} = zeros(1,5);
        empty_blob(end+1) = i+1;
    end
end
end
function [crop_boxes, size_per_crop] = target_crops(original_size, input_size)
size_per_crop = round(256/min(original_size)*original_size);
img_width = size_per_crop(2); img_height = size_per_crop(1);
% 5 crop positions
num_sizes = size(input_size, 1);
crop_boxes = zeros(size(input_size, 1) * 5, 4);

input_size = input_size(:, [2 1]);
% topleft
crop_boxes(1, 3:4) = input_size;
crop_boxes(1, 1:2) = 1 ;
% topright
crop_boxes(2, [1 4]) = [img_width-input_size(:,1)+1, input_size(:,2)];
crop_boxes(2, 2) = 1;
crop_boxes(2, 3) = img_width;
% bottomleft
crop_boxes(3, 1) = 1;
crop_boxes(3, 4) = img_height;
crop_boxes(3, 2:3) = [img_height-input_size(:,2)+1, input_size(:,1)];
% bottomright
crop_boxes(4, 3) = img_width;
crop_boxes(4, 4) = img_height;
crop_boxes(4, 1:2) = bsxfun(@minus, [img_width, img_height]+1, input_size);
% center
crop_boxes(5, 1:2) = round(bsxfun(@minus, [img_width img_height], input_size)/2);
crop_boxes(5, 3:4) = crop_boxes(4*num_sizes+1:5*num_sizes, 1:2) + input_size -1;
crop_boxes = num2cell(crop_boxes,2);
end
