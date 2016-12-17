function roidb = roidb_from_ilsvrc(imdb, varargin)
% roidb = roidb_from_voc(imdb, rootDir)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
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

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('exclude_difficult_samples',       true,   @islogical);
ip.addParamValue('with_selective_search',           false,  @islogical);
ip.addParamValue('with_edge_box',                   false,  @islogical);
ip.addParamValue('with_self_proposal',              false,  @islogical);
ip.addParamValue('rootDir',                         '.',    @ischar);
ip.addParamValue('extension',                       '',     @ischar);
ip.parse(imdb, varargin{:});
opts = ip.Results;

roidb.name = imdb.name;
if ~isempty(opts.extension)
    opts.extension = ['_', opts.extension];
end
regions_file_ss = fullfile(opts.rootDir, sprintf('/data/selective_search_data/%s%s.mat', roidb.name, opts.extension));
regions_file_eb = fullfile(opts.rootDir, sprintf('/data/edge_box_data/%s%s.mat', roidb.name, opts.extension));
regions_file_sp = fullfile(opts.rootDir, sprintf('/data/self_proposal_data/%s%s.mat', roidb.name, opts.extension));

cache_file_ss = [];
cache_file_eb = [];
cache_file_sp = [];
if opts.with_selective_search
    cache_file_ss = 'ss_';
    if~exist(regions_file_ss, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_ss);
    end
end

if opts.with_edge_box
    cache_file_eb = 'eb_';
    if ~exist(regions_file_eb, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_eb);
    end
end

if opts.with_self_proposal
    cache_file_sp = 'sp_';
    if ~exist(regions_file_sp, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_sp);
    end
end

cache_file = fullfile(opts.rootDir, ['/imdb/cache/roidb_' cache_file_ss cache_file_eb cache_file_sp imdb.name opts.extension]);
if imdb.flip
    cache_file = [cache_file '_flip'];
end
if opts.exclude_difficult_samples
    cache_file = [cache_file '_easy'];
end
cache_file = [cache_file, '.mat'];
try
    load(cache_file);
    if ~isfield('cache_file', roidb);
        roidb.cache_file = cache_file;
        save(roidb.cache_file, 'roidb');
    end
catch
    ImageNetOpt = imdb.details.ImageNetOpt;

    addpath(fullfile(ImageNetOpt.datadir, '..','evaluation'));

    roidb.name = imdb.name;
    roidb.cache_file = cache_file;
    fprintf('Loading region proposals...');
    regions = [];
    if opts.with_selective_search
        regions = load_proposals(regions_file_ss, regions);
    end
    if opts.with_edge_box
        regions = load_proposals(regions_file_eb, regions, 'edge');
    end
    if opts.with_self_proposal
        regions = load_proposals(regions_file_sp, regions, 'sp');
    end
    fprintf('done\n');
    if isempty(regions)
        fprintf('Warrning: no windows proposal is loaded !\n');
        regions.boxes = cell(length(imdb.image_ids), 1);
        if imdb.flip
            regions.images = imdb.image_ids(1:2:end);
        else
            regions.images = imdb.image_ids;
        end
    end
    xmlpath = @(i) (sprintf('%s/%s',imdb.anno_dir, [imdb.image_ids{i} '.xml']));
    if ~imdb.flip
        for i = 1:length(imdb.image_ids)
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
            try
                voc_rec = VOCreadxml(xmlpath(i));
                voc_rec = voc_rec.annotation;
            catch e
                warning(['roidb_from_ilsvrc:' e.message]);
                voc_rec = [];
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, false);
        end
    else
        for i = 1:length(imdb.image_ids)/2
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
            try
                voc_rec = VOCreadxml(xmlpath(2*i-1));
                voc_rec = voc_rec.annotation;
            catch e
                warning(['roidb_from_ilsvrc:' imdb.image_ids{i},  e.message]);
                voc_rec = [];
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i*2-1) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, false);
            roidb.rois(i*2) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, true);
        end
    end

    rmpath(fullfile(ImageNetOpt.datadir, '..','evaluation'));

    fprintf('Saving roidb to cache...');
    save(cache_file, 'roidb');
    fprintf('done\n');

end

% ------------------------------------------------------------------------
function rec = attach_proposals(voc_rec, boxes, class_to_id, exclude_difficult_samples, flip)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
if ~isempty(boxes)
    if flip
        boxes(:, [1, 3]) = str2double(voc_rec.size.width) + 1 - boxes(:, [3, 1]);
    end
end

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(voc_rec, 'object')
    common_classes = class_to_id.isKey({voc_rec.object(:).name});
    valid_objects = common_classes;
    gt_boxes = arrayfun(@(z) ...
        ([str2double(z.bndbox.xmin) str2double(z.bndbox.ymin) str2double(z.bndbox.xmax) str2double(z.bndbox.ymax)]), ...
        voc_rec.object(valid_objects),...
        'UniformOutput', false);
    gt_boxes = cell2mat(gt_boxes');
    try
        gt_classes = class_to_id.values({voc_rec.object(valid_objects).name});
    catch
        disp({voc_rec.object(valid_objects).name});
    end
    if flip && any(valid_objects)
        gt_boxes(:, [1, 3]) = str2double(voc_rec.size.width) + 1 - gt_boxes(:, [3, 1]);
    end
    all_boxes = cat(1, gt_boxes, boxes);
    gt_classes = cat(1, gt_classes{:});
    num_gt_boxes = size(gt_boxes, 1);
else
    gt_boxes = [];
    all_boxes = boxes;
    gt_classes = [];
    num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

num_gt_class = max(cell2mat(class_to_id.values));
rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, num_gt_class, 'single');
for i = 1:num_gt_boxes
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions, method)
% ------------------------------------------------------------------------
if nargin < 3
    method = 'selective';
end
switch method
    case 'selective'
        % change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
        postprocess = @(bbs) bbs(:, [2 1 4 3]);
    case 'edge'
        % change edge box from [x1 y1 w h] to [x1 y1 x2 y2]
        postprocess = @(bbs_all) cellfun(@(bbs) ...
            cat(2, bbs(:, [1 2]), bbs(:, [1 2]) + bbs(:, [3,4])), bbs_all, 'UniformOutput', false);
    case 'sp'

    otherwise
        error('proposal method is one of ''edge'', ''selective'', ''sp''\n');
end
if isempty(regions)
    regions = load(proposal_file);
    regions.boxes = postprocess(regions.boxes);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = postprocess(regions.boxes);
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes(:), regions_more.boxes(:), 'UniformOutput', false);
end
