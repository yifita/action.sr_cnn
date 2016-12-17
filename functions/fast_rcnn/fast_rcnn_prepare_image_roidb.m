function [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, varargin)
% [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
%   Gather useful information from imdb and roidb
%   pre-calculate mean (bbox_means) and std (bbox_stds) of the regression
%   term for normalization
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
ip = inputParser;
ip.addRequired('conf', @isstruct);
ip.addRequired('imdbs', @iscell);
ip.addRequired('roidbs', @iscell);
ip.addParameter('bbox_means', [], @ismatrix);
ip.addParameter('bbox_stds', [], @ismatrix);
ip.addParameter('sub_ind', [], @iscell);
ip.parse(conf, imdbs, roidbs, varargin{:});
opts = ip.Results;
bbox_means = opts.bbox_means;
bbox_stds = opts.bbox_stds;

if isempty(opts.sub_ind)
    opts.sub_ind = cellfun(@(x) reshape(1:length(x.image_ids), [], 1), imdbs, 'uni', false);
elseif iscell(opts.sub_ind{1})
    % make sure sub_ind is a cell array of ind arrays
    longer_dim = find(size(opts.sub_ind{1}{1}) > 1);
    opts.sub_ind = cellfun(@(x) cat(longer_dim, x{:}), opts.sub_ind, 'uni', false);
end
assert(all(cellfun(@isnumeric, opts.sub_ind)));
image_roidb = ...
    cellfun(@(x, y, ind) ... // @(imdbs, roidbs)
            arrayfun(@(z) ... //@([1:length(x.image_ids)])
                    struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                    'overlap', y.rois(z).overlap, 'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
            ind, 'UniformOutput', true),...
    imdbs, roidbs, opts.sub_ind, 'UniformOutput', false);
image_roidb = cat(1, image_roidb{:});

% enhance roidb to contain bounding-box regression targets
append_bbox_regression_targets();

function append_bbox_regression_targets()
    % bbox_means and bbox_stds -- (k+1) * 4, include background class

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    valid_imgs = true(num_images, 1);
    for i = 1:num_images
       rois = image_roidb(i).boxes;
       [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
           compute_targets(conf, rois, image_roidb(i).overlap);
    end
    if ~all(valid_imgs)
        image_roidb = image_roidb(valid_imgs);
        num_images = length(image_roidb);
        fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
    end

    if ~(~isempty(bbox_means) && ~isempty(bbox_stds))
        fprintf('fast_rcnn_prepare_image_roidb: Calculating bbox_means bbox_stds...\n');
        % Compute values needed for bbox_means and bbox_stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(num_classes, 1) + eps;
        sums = zeros(num_classes, 4);
        squared_sums = zeros(num_classes, 4);
        for i = 1:num_images
           targets = image_roidb(i).bbox_targets;
           for cls = 1:num_classes
              cls_inds = find(targets(:, 1) == cls);
              if ~isempty(cls_inds)
                 class_counts(cls) = class_counts(cls) + length(cls_inds);
                 sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
                 squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
              end
           end
        end

        bbox_means = bsxfun(@rdivide, sums, class_counts);
        bbox_stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), bbox_means.^2)).^0.5;

        % add background class
        bbox_means = [0, 0, 0, 0; bbox_means];
        bbox_stds = [0, 0, 0, 0; bbox_stds];
    end

    % Normalize targets
    for i = 1:num_images
        targets = image_roidb(i).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), bbox_means(cls+1, :));
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), bbox_stds(cls+1, :));
            end
        end
    end
end
end

function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)

    overlap = full(overlap);

    [max_overlaps, max_labels] = max(overlap, [], 2);

    % ensure ROIs are floats
    rois = single(rois);

    bbox_targets = zeros(size(rois, 1), 5, 'single');

    % Indices of ground-truth ROIs
    gt_inds = find(max_overlaps == 1);

    if ~isempty(gt_inds)
        % Indices of examples for which we try to make predictions
        ex_inds = find(max_overlaps >= conf.bbox_thresh);

        % Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));

        assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));

        % Find which gt ROI each ex ROI has max overlap with:
        % this will be the ex ROI's gt target
        [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
        gt_rois = rois(gt_inds(gt_assignment), :);
        ex_rois = rois(ex_inds, :);

        [regression_label] = fast_rcnn_bbox_transform(ex_rois, gt_rois);

        bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
    end

    % Select foreground ROIs as those with >= fg_thresh overlap
    is_fg = max_overlaps >= conf.fg_thresh;
    % Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
    is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;

    % check if there is any fg or bg sample. If no, filter out this image
    is_valid = true;
    if ~any(is_fg | is_bg)
        is_valid = false;
    end
end
