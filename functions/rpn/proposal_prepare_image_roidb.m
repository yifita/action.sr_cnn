function [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb(conf, imdbs, roidbs, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

if ~exist('bbox_means', 'var')
    bbox_means = [];
    bbox_stds = [];
end

if conf.target_only_gt
    image_roidb = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
        arrayfun(@(z) ... //@([1:length(x.image_ids)])
        struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, 'num_classes', x.num_classes, ...
        'boxes', y.rois(z).boxes(y.rois(z).gt, :), 'class', y.rois(z).class(y.rois(z).gt, :), 'image', [], 'bbox_targets', []), ...
        [1:length(y.rois)]', 'UniformOutput', true),...
        imdbs, roidbs, 'UniformOutput', false);
else
    image_roidb = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
        arrayfun(@(z) ... //@([1:length(x.image_ids)])
        struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
        'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
        [1:length(y.rois)]', 'UniformOutput', true),...
        imdbs, roidbs, 'UniformOutput', false);
end
image_roidb = cat(1, image_roidb{:});

% enhance roidb to contain bounding-box regression targets
append_bbox_regression_targets();


    function append_bbox_regression_targets()
        % bbox_means and bbox_stds -- (k+1) * 4, include background class

        num_images = length(image_roidb);
        for i = 1:num_images
            % for fcn, anchors are concated as [channel, height, width], where channel is the fastest dimension.
            [anchors, im_scales] = proposal_locate_anchors(conf, image_roidb(i).im_size); %TODO

            gt_rois = image_roidb(i).boxes;
            gt_labels = image_roidb(i).class;
            im_size = image_roidb(i).im_size;
            image_roidb(i).bbox_targets = cellfun(@(x, y) ...
                compute_targets(conf, scale_rois(gt_rois, im_size, y), gt_labels,  x, image_roidb(i), y), ...
                anchors, im_scales, 'UniformOutput', false);
        end

        fprintf('Computing means and stds...');
        if ~(exist('bbox_means', 'var') && ~isempty(bbox_means) && exist('bbox_stds', 'var') && ~isempty(bbox_stds))
            % Compute values needed for bbox_means and bbox_stds
            % var(x) = E(x^2) - E(x)^2
            class_counts = zeros(1, 1) + eps;
            sums = zeros(1, 4);
            squared_sums = zeros(1, 4);
            for i = 1:num_images
                for j = 1:length(conf.scales)
                    targets = image_roidb(i).bbox_targets{j};
                    gt_inds = find(targets(:, 1) > 0);
                    if ~isempty(gt_inds)
                        class_counts = class_counts + length(gt_inds);
                        sums = sums + sum(targets(gt_inds, 2:end), 1);
                        squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
                    end
                end
            end

            bbox_means = bsxfun(@rdivide, sums, class_counts);
            bbox_stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), bbox_means.^2)).^0.5;
        end
        fprintf('done\n');

        % Normalize targets
        fprintf('Normalizing targets...');
        for i = 1:num_images
            for j = 1:length(conf.scales)
                targets = image_roidb(i).bbox_targets{j};
                gt_inds = find(targets(:, 1) > 0);
                if ~isempty(gt_inds)
                    image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
                        bsxfun(@minus, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), bbox_means);
                    image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
                        bsxfun(@rdivide, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), bbox_stds);
                end
            end
        end
        fprintf('done\n');
    end
end


function scaled_rois = scale_rois(rois, im_size, im_scale)
scaled_rois = [];
if isempty(rois)
    return;
end
im_size_scaled = round(im_size * im_scale);
scale = (im_size_scaled - 1) ./ (im_size - 1);
scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end

function bbox_targets = compute_targets(conf, gt_rois, gt_labels, ex_rois, image_roidb, im_scale)
% output: bbox_targets
%   positive: [class_label, regression_label]
%   ingore: [0, zero(regression_label)]
%   negative: [-1, zero(regression_label)]

if isempty(gt_rois)
    bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
    bbox_targets(:, 1) = -1;
    return;
end

% ensure gt_labels is in single
gt_labels = single(gt_labels);
assert(all(gt_labels > 0));

% calc overlap between ex_rois(anchors) and gt_rois
ex_gt_overlaps = boxoverlap(ex_rois, gt_rois);

% drop anchors which run out off image boundaries, if necessary
if conf.drop_boxes_runoff_image
    contained_in_image = is_contain_in_image(ex_rois, round(image_roidb.im_size * im_scale));
    ex_gt_overlaps(~contained_in_image, :) = 0;
end

% for each ex_rois(anchors), get its max overlap with all gt_rois
[ex_max_overlaps, ex_assignment] = max(ex_gt_overlaps, [], 2);

% for each gt_rois, get its max overlap with all ex_rois(anchors), the
% ex_rois(anchors) are recorded in gt_assignment
% gt_assignment will be assigned as positive
% (assign a rois for each gt at least)
[gt_max_overlaps, gt_assignment] = max(ex_gt_overlaps, [], 1);

% ex_rois(anchors) with gt_max_overlaps maybe more than one, find them
% as (gt_best_matches)
[gt_best_matches, gt_ind] = find(bsxfun(@eq, ex_gt_overlaps, [gt_max_overlaps]));

% Indices of examples for which we try to make predictions
% both (ex_max_overlaps >= conf.fg_thresh) and gt_best_matches are
% assigned as positive examples
fg_inds = unique([find(ex_max_overlaps >= conf.fg_thresh); gt_best_matches]);

% Indices of examples for which we try to used as negtive samples
% the logic for assigning labels to anchors can be satisfied by both the positive label and the negative label
% When this happens, the code gives the positive label precedence to
% pursue high recall
bg_inds = setdiff(find(ex_max_overlaps < conf.bg_thresh_hi & ex_max_overlaps >= conf.bg_thresh_lo), ...
    fg_inds);

if conf.drop_boxes_runoff_image
    contained_in_image_ind = find(contained_in_image);
    fg_inds = intersect(fg_inds, contained_in_image_ind);
    bg_inds = intersect(bg_inds, contained_in_image_ind);
end

% Find which gt ROI each ex ROI has max overlap with:
% this will be the ex ROI's gt target
target_rois = gt_rois(ex_assignment(fg_inds), :);
src_rois = ex_rois(fg_inds, :);

% we predict regression_label which is generated by an un-linear
% transformation from src_rois and target_rois
[regression_label] = fast_rcnn_bbox_transform(src_rois, target_rois);

bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
bbox_targets(fg_inds, :) = [gt_labels(ex_assignment(fg_inds)), regression_label];
bbox_targets(bg_inds, 1) = -1;

if 0 % debug
    %%%%%%%%%%%%%%
    im = imread(image_roidb.image_path);
    [im, im_scale] = prep_im_for_blob(im, conf.image_means, conf.scales, conf.max_size);
    imshow(mat2gray(im));
    hold on;
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), ...
        num2cell(src_rois, 2));
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g'), ...
        num2cell(target_rois, 2));
    hold off;
    %%%%%%%%%%%%%%
end

bbox_targets = sparse(bbox_targets);
end

function contained = is_contain_in_image(boxes, im_size)
contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);

contained = all(contained, 2);
end
