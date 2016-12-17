function input_blobs = get_test_minibatch(image_roidb, conf, varargin)
% GET_TEST_MINIBATCH create a test minibatch using the frame+roi generated from
% prepare_minibatch
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('image_roidb',           @isstruct);
ip.addRequired('conf',                  @isstruct);
ip.addParameter('crop',                 true,       @islogical);
ip.addParameter('flip',                 true,       @islogical);
% threshold person bb aoi to drop a sample
ip.addParameter('min_inter_ratio',      0.5,     @isscalar);
ip.addParameter('min_rois_length',      20,         @isscalar);
ip.addParameter('max_num_person',       2,          @isscalar);
ip.addParameter('label',                false,      @islogical);
ip.addParameter('upscale',              1.2,        @isscalar);
ip.addParameter('batchsize',            16,         @isscalar);
ip.parse(image_roidb, conf, varargin{:});
opts = ip.Results;

assert(conf.n_person <= 1, 'only support 1 person per image');
num_channels = sum([conf.n_person > 0, conf.obj_per_img > 0, conf.use_scene]);
do_merge = isfield(conf, 'merge') && num_channels > 1;

num_images = length(image_roidb);
if opts.flip
    num_images = num_images*2;
end

%% initialize
im_blob = cell(num_images, 1);
if conf.n_person > 0
    person_blob = cell(num_images, 1);
end
if conf.obj_per_img > 0
    obj_blob = cell(num_images, 1);
    size_blob = cell(num_images, 1);
end
if opts.label
    label_blob = cell(num_images, 1);
end

    %% return empty as error handler
    function output = returnEmpty(s,varargin)
            output = [];
    end
    %% returnSceneRoi: return scene rois as error handler
    function output = returnSceneRoi(s, varargin)
        output = [1 1 conf.input_size(2) conf.input_size(1)];
    end
    %% returnDummyRoi: return scene rois as error handler
    function output = returnDummyRoi(s, varargin)
        output = [0 0 0 0];
    end
    %% flip_rois: flip rois [batch_ind x0 y0 x1 y1]
    function rois = flip_rois(rois)
        rois(:, [4 2]) = conf.input_size(2)-rois(:,[2 4])+1;
    end
if opts.crop
    [crops, size_per_crop] = target_crops(conf.input_size);
end

counter_batch = 0;
for i = 1:length(image_roidb)
    % image blob
    [im, im_scale] = get_image_blob();
    % number of found relevant people in a crop area
    sample_in_crop = ones(length(im), 1);
    % person rois blob
    if conf.n_person > 0
        try
            % could get duplicate bounding boxes due to intercepting tubes
            person_rois = unique(image_roidb(i).person_rois(:,1:4), 'rows');
            if opts.upscale ~= 1
                person_center = [person_rois(:,3)+person_rois(:,1) person_rois(:,4)+person_rois(:,2)]/2;
                person_rois = (person_rois - person_center(:,[1 2 1 2])) * opts.upscale + person_center(:,[1 2 1 2]);
                person_rois(:,1) = max(person_rois(:,1), 1);
                person_rois(:,2) = max(person_rois(:,2), 1);
                person_rois(:,3) = min(person_rois(:,3), image_roidb(i).size(2));
                person_rois(:,4) = min(person_rois(:,4), image_roidb(i).size(1));
            end
            if opts.crop
                person_rois = fast_rcnn_map_im_rois_to_feat_rois(...
                    [], person_rois, size_per_crop./image_roidb(i).size);
                inters_valid = cellfun(@(crop) boxinter(person_rois, crop, opts.min_inter_ratio), crops, 'uni', false);
                % get cropped people bounding box. 
                person_rois = cellfun(@(inter, crop, scale) ...
                    get_cropped_rois(person_rois(inter, :), crop, scale), ...
                    inters_valid, crops, im_scale, 'uni', false, 'ErrorHandler', @returnSceneRoi);
            else
                person_rois = cellfun(@(scale) fast_rcnn_map_im_rois_to_feat_rois(...
                    [], person_rois, scale), im_scale, 'uni', false);
            end
            [~, person_rois] = cellfun(@(x) ...
                check_rois_pool(x, opts.min_rois_length, conf.min_bb_length, conf.input_size), person_rois, 'uni', false);
            % crops that don't contain valid person bounding box
            tmp = cellfun('isempty', person_rois);
            if all(tmp)
                error('no valid person')
            end
            person_rois(tmp) = {nan(1,4)};
        catch
            person_rois = cell(size(im_scale));
            [person_rois{:}] = deal([1 1 conf.input_size(2) conf.input_size(1)]);
        end
        sample_in_crop = cellfun(@(x) size(x, 1), person_rois);
    end
    batch_ind = mat2cell([counter_batch+1:counter_batch+sum(sample_in_crop)]', sample_in_crop);
    counter_batch = counter_batch + sum(sample_in_crop);
    sample_in_crop_cell = num2cell(sample_in_crop, 2);
    % objects rois blob [class_id x1 y1 x2 y2 cls_score]
    if conf.obj_per_img > 0
        try
            obj_rois = image_roidb(i).obj_rois(:,2:5);
            if opts.crop
                obj_rois = fast_rcnn_map_im_rois_to_feat_rois(...
                    [], obj_rois, size_per_crop./image_roidb(i).size);
                inters = cellfun(@(crop) boxinter(obj_rois, crop, opts.min_inter_ratio), crops, 'uni', false);
                obj_rois = cellfun(@(inter, crop, scale) ...
                    get_cropped_rois(obj_rois(inter, :), crop, scale), ...
                    inters, crops, im_scale, 'uni', false, 'ErrorHandler', @returnEmpty);
            else
                obj_rois = cellfun(@(scale) fast_rcnn_map_im_rois_to_feat_rois(...
                    [], obj_rois, scale), im_scale, 'uni', false);
            end
            % stricter filter rule (min_rois_length) for obj_rois than for person_rois
            [valid, obj_rois] = cellfun(@(x) ...
                check_rois_pool(x, opts.min_rois_length, conf.min_bb_length, conf.input_size), obj_rois, 'uni', false);
            obj_rois = cellfun(@(x, inds) x(inds,:), obj_rois, valid, 'uni', false);
            if conf.n_person > 0 && ~isempty(person_rois)
                inter = cellfun(@(obj, per) boxinter(obj, per, 0), obj_rois, person_rois, 'uni',false);
                obj_rois = cellfun(@(obj,i) obj(all(i, 2), :), obj_rois, inter, 'uni', false);
            end
            obj_rois = cellfun(@(x) x(1:min(conf.obj_per_img, size(x,1)),:), obj_rois, 'uni', false, 'ErrorHandler', @returnEmpty);
            % fill empty rois with scene rois
            empty_rois = cellfun('isempty', obj_rois);
            obj_rois(empty_rois) = {[1 1 conf.input_size(2) conf.input_size(1)]};
        catch
            obj_rois = cell(size(im_scale));
            [obj_rois{:}] = deal([1 1 conf.input_size(2) conf.input_size(1)]);
        end
        obj_num = cellfun(@(x) size(x, 1), obj_rois, 'uni', false);
    end
    % duplicate obj_blob for each found person
    if conf.n_person > 0
        person_rois = cellfun(@(ind, rois) [ind rois], batch_ind, person_rois, 'uni', false);
        person_blob{i} = cat(1, person_rois{:});
    end
    if conf.obj_per_img > 0
        obj_batch_ind = cellfun(@(inds, num) cell2mat(arrayfun(@(x) ...
                    x*ones(num, 1), inds, 'uni', false)), batch_ind, obj_num, 'uni', false);
        obj_num = cellfun(@(x, y) repmat(x, [y,1]), obj_num, sample_in_crop_cell, 'uni', false);
        obj_rois = cellfun(@(x, y) repmat(x, [y,1]), obj_rois, sample_in_crop_cell, 'uni', false);
        obj_rois = cellfun(@(x, y) [x y], obj_batch_ind, obj_rois, 'uni', false);
        obj_blob{i} = cat(1, obj_rois{:});
        size_blob{i} = cat(1, obj_num{:});
    end
    im = cellfun(@(x, y) repmat(x, [1 1 1 y]), im, sample_in_crop_cell, 'uni', false);
    im_blob{i} = cat(4, im{:});
    if opts.label
        label = cellfun(@(y) repmat(image_roidb(i).class_id-1, y, 1), sample_in_crop_cell, 'uni', false);
        label_blob{i} = cat(1, label{:});
    end
end
if opts.flip
    im_blob(length(image_roidb)+1:num_images) = ...
    cellfun(@fliplr, im_blob(1:length(image_roidb)),'uni', false);
    if conf.n_person > 0
        person_blob(length(image_roidb)+1:num_images) = ...
        cellfun(@flip_rois, person_blob(1:length(image_roidb)), 'uni', false);
        person_blob(length(image_roidb)+1:num_images) = ...
        cellfun(@(x) [x(:,1)+counter_batch x(:,2:end)], person_blob(length(image_roidb)+1:num_images), 'uni', false);
    end
    if conf.obj_per_img > 0
        obj_blob(length(image_roidb)+1:num_images) = ...
        cellfun(@flip_rois, obj_blob(1:length(image_roidb)), 'uni', false);
        obj_blob(length(image_roidb)+1:num_images) = ...
                cellfun(@(x) [x(:,1)+counter_batch x(:,2:end)], obj_blob(length(image_roidb)+1:num_images), 'uni', false);
        size_blob(length(image_roidb)+1:num_images) = size_blob(1:length(image_roidb));
    end
    if opts.label
        label_blob(length(image_roidb)+1:num_images) = label_blob(1:length(image_roidb));
    end
end
%% divide to fit batchsize
% permute data into caffe c++ memory, thus [num, channels, height, width]
im_blob = cat(4, im_blob{:}); % from rgb to brg
num_blobs = ceil(size(im_blob,4)/opts.batchsize);
total_batches = size(im_blob,4);

im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
im_blob = arrayfun(@(i) im_blob(:,:,:,(i-1)*opts.batchsize+1:min(total_batches, i*opts.batchsize)),1:num_blobs,'uni',false);
im_blob = cellfun(@(x) single(permute(x, [2, 1, 3, 4])), im_blob, 'uni', false);
im_blob = reshape(im_blob, [], 1);
ignore_blob = cellfun(@(x) ones(num_channels, size(x, 4), 'single'), im_blob, 'uni', false);
if conf.n_person > 0
    person_blob = cat(1, person_blob{:}); % from rgb to brg
    person_blob = person_blob - 1; % to c's index (start from 0)
    person_blob = divide_rois_blob(person_blob, num_blobs, opts.batchsize);
    dummy_batch = cellfun(@(x) x(all(isnan(x(:,2:5)),2),1), person_blob, 'uni', false);
    for i = 1:num_blobs
        ignore_blob{i}(conf.PERSON_BLOB_IDX+1, dummy_batch{i}+1) = 0;
    end
    person_blob = cellfun(@(x) single(permute(x, [3, 4, 2, 1])), person_blob, 'uni',false);
end
if conf.obj_per_img > 0
    obj_blob = cat(1, obj_blob{:}); % from rgb to brg
    obj_blob = obj_blob - 1; % to c's index (start from 0)
    obj_blob = divide_rois_blob(obj_blob, num_blobs, opts.batchsize);
    dummy_batch = cellfun(@(x) x(ismember(x(:,2:5), zeros(1,4,'single'), 'rows'),1), obj_blob, 'uni', false);
    for i = 1:num_blobs
        ignore_blob{i}(conf.OBJ_BLOB_IDX+1, dummy_batch{i}+1) = 0;
    end
    obj_blob = cellfun(@(x) single(permute(x, [3, 4, 2, 1])), obj_blob, 'uni', false);
    size_blob = cat(1, size_blob{:});
    size_blob = arrayfun(@(i) size_blob((i-1)*opts.batchsize+1:min(total_batches, i*opts.batchsize)),1:num_blobs,'uni',false);
    size_blob = cellfun(@(x) single(permute(x, [3, 4, 2, 1])), size_blob,'uni',false);
    size_blob = reshape(size_blob, [], 1);
end
ignore_blob = cellfun(@(x) permute(x, [2 1 3 4]), ignore_blob, 'uni', false);
if conf.use_scene
    scene_rois = repmat([1 1 conf.input_size(2) conf.input_size(1)], total_batches, 1);
    scene_blob = [reshape(1:total_batches, [], 1) scene_rois];
    scene_blob = scene_blob - 1; % to c's index (start from 0)
    scene_blob = divide_rois_blob(scene_blob, num_blobs, opts.batchsize);
    scene_blob = cellfun(@(x) single(permute(x, [3, 4, 2, 1])), scene_blob, 'uni', false);
end
if conf.use_scene && conf.n_person == 0 && conf.obj_per_img == 0 % only scene
    input_blobs = [im_blob];
elseif conf.n_person > 0 && conf.obj_per_img == 0 && ~conf.use_scene % only person
    input_blobs =[im_blob, person_blob];
elseif conf.n_person == 0 && conf.obj_per_img > 0 && ~conf.use_scene % only object
    input_blobs = [im_blob, obj_blob, size_blob];
elseif do_merge && conf.n_person > 0 && conf.obj_per_img == 0 && conf.use_scene % person + scene
    input_blobs = [im_blob, person_blob, scene_blob, ignore_blob];
elseif do_merge && conf.n_person > 0 && conf.obj_per_img > 0 && ~conf.use_scene % object + person
    input_blobs = [im_blob, person_blob, obj_blob, size_blob];
elseif do_merge && conf.n_person == 0 && conf.obj_per_img > 0 && conf.use_scene % object + scene
    input_blobs = [im_blob, scene_blob, obj_blob, size_blob];
elseif do_merge && conf.n_person > 0 && conf.obj_per_img > 0 && conf.use_scene % all
    input_blobs = [im_blob, person_blob, scene_blob, obj_blob, size_blob, ignore_blob ];
end
if opts.label
    label_blob = cat(1,label_blob{:});
    label_blob = arrayfun(@(i) label_blob((i-1)*opts.batchsize+1:min(total_batches, i*opts.batchsize)),1:num_blobs,'uni',false);
    label_blob = cellfun(@(x) single(permute(x, [3 4 2 1])), label_blob, 'uni',false);
    input_blobs = horzcat(input_blobs, reshape(label_blob, [], 1));
end
%% Build an input blob from the images in the roidb at the specified scales.
    function [im, im_scale] = get_image_blob()
        % im_scale net_input_size/cropped_size or net_input_size/original_size (if rng_crop == false)
        im = single(imread(image_roidb(i).frame_path));
        im = bsxfun(@minus, im, conf.image_means);
        if ~opts.crop
            im_scale = {conf.input_size./size(im(:,:,1))};
            im = {imresize(im, conf.input_size, 'bilinear', 'antialiasing', false)};
        else
            if ~isequal(size_per_crop, size(im(:,:,1)))
                im = imresize(im , size_per_crop, 'bilinear', 'antialiasing', false);
            end
            % crop
            im = cellfun(@(crop) im(crop(2):crop(4), crop(1):crop(3), :), crops, 'uni', false);
            im_scale = cellfun(@(x) conf.input_size./size(x(:,:,1)), im, 'uni', false);
        end
    end

end

%% Check rois according to pooling size, if required enlarge rois
function [valid, rois] = check_rois_pool(rois, min_rois_length, min_bb_length, input_size)
valid = [];
if isempty(rois)
    return
end
w = rois(:,3)-rois(:,1)+1;
h = rois(:,4)-rois(:,2)+1;
valid = (w >= min_rois_length) & (h >= min_rois_length);
if nargout > 1
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
rois(valid,:) = fast_rcnn_map_im_rois_to_feat_rois([], rois(valid,:), im_scale_to_target);
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
function [crop_boxes, size_per_crop] = target_crops(input_size)
size_per_crop = [256 340];
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
assert(max(crop_boxes(:,3)) <= size_per_crop(2)); assert(max(crop_boxes(:,4)) <= size_per_crop(1));
crop_boxes = num2cell(crop_boxes,2);
end
