function imdb = imdb_from_voc(root_dir, image_set, year, imdb_cache_dir, flip)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the PASCAL VOC devkit located
%   at root_dir using the image_set and year.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

if nargin < 5
	flip = false;
end

cache_file = fullfile(imdb_cache_dir, ['imdb/cache/imdb_voc_' year '_' image_set]);
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
	VOCopts = get_voc_opts(root_dir);
	VOCopts.testset = image_set;

	imdb.name = ['voc_' year '_' image_set];
	imdb.cache_file = cache_file;
	imdb.image_dir = fileparts(VOCopts.imgpath);
	imdb.anno_dir = fileparts(VOCopts.annopath);
	imdb.image_ids = textread(sprintf(VOCopts.imgsetpath, image_set), '%s');
	imdb.extension = 'jpg';
	imdb.flip = flip;
	if flip
		image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
		flip_image_at = @(i) sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
		for i = 1:length(imdb.image_ids)
			if ~exist(flip_image_at(i), 'file')
				im = imread(image_at(i));
				imwrite(fliplr(im), flip_image_at(i));
			end
		end
		img_num = length(imdb.image_ids)*2;
		image_ids = imdb.image_ids;
		imdb.image_ids(1:2:img_num) = image_ids;
		imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
	end
	% NOTE: extend classes to ILSVRC
	imagenet_meta = Dataset.script_get_wnid;
	class_to_id = Dataset.get_classID_map(imagenet_meta);
	imdb.classes = intersect(class_to_id.keys, VOCopts.classes);
	imdb.class_to_id = class_to_id;
	imdb.num_classes = length(imdb.classes);
	imdb.class_ids = cell2mat(imdb.class_to_id.values(imdb.classes));

	% private VOC details
	imdb.details.VOCopts = VOCopts;

	% VOC specific functions for evaluation and region of interest DB
	imdb.eval_func = @imdb_eval_voc;
	imdb.roidb_func = @roidb_from_voc;

	% Save jpg size
    	valid_imgs = true(length(imdb.image_ids), 1);
    	imdb.sizes = nan(length(imdb.image_ids), 2);
	image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
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
	assert(length(imdb.image_ids) == size(imdb.sizes,1) && ~any(isnan(imdb.sizes(:))));

	imdb.image_at = @(i) ...
	    sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
	fprintf('%d images in %s\n',length(imdb.image_ids), imdb.name);
	fprintf('Saving imdb to cache...');
	save(cache_file, 'imdb');
	fprintf('done\n');
end
