function imdb = imdb_from_ilsvrc(root_dir, image_set, flip, imdb_cache_dir, N)
% imdb = imdb_from_ilsvrc(root_dir, image_set)
%   Builds an image database for ILSVRC dataset
%
%   Modified from Faster-RCNN detection code

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%imdb.name = 'ilsvrc_2014_train'
%imdb.image_dir = '/import/mfs/ait/Yifan/VOCdevkit/VOC2007/JPEGImages/'
%imdb.anno_dir = '/import/mfs/ait/Yifan/VOCdevkit/VOC2007/Annotations'
%imdb.extension = 'jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

if nargin < 5
    if strcmp(image_set, 'train')
	N = 600;
    end
end
cache_file = fullfile(imdb_cache_dir, ['imdb/cache/imdb_ilsvrc_2014_' image_set]);
if flip
    cache_file = [cache_file, '_flip'];
end
try
    load(cache_file);
    if ~isfield(imdb, 'cache_file')
        imdb.cache_file = cache_file;
        save(imdb.cache_file, 'imdb');
    end
catch
    imdb.name = ['ilsvrc_2014_' image_set];
    imdb.cache_file = cache_file;

    % ILSVRC data
    imagenet_meta = Dataset.script_get_wnid;

    ImageNetOpt.datadir = fullfile(root_dir, 'data');
    ImageNet_ids = {};
    if strfind(image_set, 'train')
        ImageNetOpt.imgsetpath = cell(2,1);
        ImageNetOpt.imgsetpath{1} = fullfile(ImageNetOpt.datadir, 'det_lists', 'train_pos_%d.txt');
        ImageNetOpt.imgsetpath{2} = fullfile(ImageNetOpt.datadir, 'det_lists', 'train_neg_%d.txt');
        ImageNet_ids = cell(length(imagenet_meta.toplevel_idx)*2*N,1);
        counter = 0;
        for i = 1:length(imagenet_meta.toplevel_idx)
            for s = 1:length(ImageNetOpt.imgsetpath)
                % smaller sample from negative set
                tmp = textread(sprintf(...
                    ImageNetOpt.imgsetpath{s}, imagenet_meta.toplevel_idx(i)), '%s', ceil(N/(s^(2-1))));
                ImageNet_ids(counter+1:counter+length(tmp)) = tmp;
                counter = counter + length(tmp);
            end
        end
        ImageNet_ids = ImageNet_ids(1:counter,:);

        imdb.image_dir = fullfile(root_dir, 'data', 'ILSVRC2014_DET_train');
        imdb.anno_dir = fullfile(ImageNetOpt.datadir, 'ILSVRC2014_DET_bbox_train');
    elseif strfind(image_set, 'val')
        ImageNetOpt.imgsetpath = fullfile(ImageNetOpt.datadir, 'det_lists', [image_set '.txt']);
        [ImageNet_ids, ~] = textread(ImageNetOpt.imgsetpath, '%s %d');
        ImageNet_ids = ImageNet_ids(1:2:end);
	    ImageNet_ids = ImageNet_ids(1:5:end);
        imdb.image_dir = fullfile(root_dir, 'data', 'ILSVRC2013_DET_val');
        imdb.anno_dir = fullfile(ImageNetOpt.datadir, 'ILSVRC2013_DET_bbox_val');
    elseif strfind(image_set, 'test')
        % NOTE: still use val but all of them
        ImageNetOpt.imgsetpath = fullfile(ImageNetOpt.datadir, 'det_lists', ['val.txt']);
        [ImageNet_ids, ~] = textread(ImageNetOpt.imgsetpath, '%s %d');
        imdb.image_dir = fullfile(root_dir, 'data', 'ILSVRC2013_DET_val');
        imdb.anno_dir = fullfile(ImageNetOpt.datadir, 'ILSVRC2013_DET_bbox_val');
    end
    imdb.image_ids = ImageNet_ids;
    imdb.extension = 'JPEG';

    image_at = @(i) sprintf('%s%s%s.%s',imdb.image_dir, filesep, imdb.image_ids{i}, imdb.extension);
    imdb.flip = flip;
    if flip
        for i = 1:length(ImageNet_ids)
            image_path = image_at(i);
            if ~exist(image_path,'file')
                continue;
            end
            if ~exist(append_flip(image_path), 'file')
                im = imread(image_path);
                imwrite(fliplr(im), append_flip(image_path));
            end
        end
        img_num = length(imdb.image_ids)*2;
        imdb.image_ids = cell(img_num,1);
        imdb.image_ids(1:2:img_num) = ImageNet_ids;
        imdb.image_ids(2:2:img_num) = strcat(ImageNet_ids, '_flip');
    end

    % extend classes to ILSVRC
    imdb.classes = keys(imagenet_meta.ImageNet_toplevel);
    imdb.class_to_id = Dataset.get_classID_map(imagenet_meta);
    imdb.num_classes = imagenet_meta.ImageNet_toplevel.Count;
    imdb.class_ids = cell2mat(imdb.class_to_id.values(imagenet_meta.ImageNet_toplevel.values));

    % private VOC details
    imdb.details.ImageNetOpt = ImageNetOpt;

    % VOC specific functions for evaluation and region of interest DB
    % TODO: eval_func
    imdb.eval_func = @imdb_eval_voc;
    imdb.roidb_func = @roidb_from_ilsvrc;

    % Save jpg size
    image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
    valid_imgs = true(length(imdb.image_ids), 1);
    imdb.sizes = nan(length(imdb.image_ids), 2);
    for i = 1:length(imdb.image_ids)
        try
            tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
            info = imfinfo(image_at(i));
            if (info.Height < 100 || info.Width < 100)
                error('Image is too small');
            end
            if (strcmp(info.ColorType, 'grayscale'))
                error('Image is not rgb');
            end
            imdb.sizes(i, :) = [info.Height info.Width];
        catch e
            valid_imgs(i) = false;
            fprintf('%s: %s\n', image_at(i), e.message);
        end
    end
    if flip
    	invalid_idx = find(~valid_imgs);
    	invalid_flipped_idx = invalid_idx(mod(invalid_idx, 2)==0);
    	invalid_org_idx = invalid_idx(mod(invalid_idx, 2)==1);
    	valid_imgs(invalid_flipped_idx - 1) = false;
    	valid_imgs(invalid_org_idx + 1) = false;
    	img_num = length(find(valid_imgs));
    	assert(mod(img_num,2) == 0);
    end

    imdb.image_ids = imdb.image_ids(valid_imgs);
    imdb.sizes = imdb.sizes(valid_imgs,:);
    assert(size(imdb.sizes,1) == length(imdb.image_ids) && all(~isnan(imdb.sizes(:))));
    imdb.image_at = @(i) sprintf('%s%s%s.%s',imdb.image_dir, filesep, imdb.image_ids{i}, imdb.extension);
    fprintf('Saving imdb to cache...');
    save(cache_file, 'imdb');
    fprintf('done\n');
end

%% append_flip: function description
function flip_image_path = append_flip(image_path)
[filedir, filename, extension] = fileparts(image_path);
flip_image_path = fullfile(filedir, sprintf('%s_flip%s', filename, extension));
