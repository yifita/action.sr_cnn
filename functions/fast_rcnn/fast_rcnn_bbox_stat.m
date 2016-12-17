function [bbox_means, bbox_stds] = fast_rcnn_bbox_stat(conf, roidbs)
% get means and stds of boxes for each class

num_classes = size(roidbs{1}.rois(1).overlap, 2);
class_counts = zeros(num_classes, 1) + eps;
sums = zeros(num_classes, 4);
squared_sums = zeros(num_classes, 4);
for j = 1:length(roidbs)
    for i = 1:length(roidbs{j}.rois)
        rois = roidbs{j}.rois(i).boxes;
        [bbox_targets, valid_img] = ...
            compute_targets(conf, rois, roidbs{j}.rois(i).overlap);
        if valid_img
            for cls = 1:num_classes
                cls_inds = find(bbox_targets(:, 1) == cls);
                if ~isempty(cls_inds)
                    class_counts(cls) = class_counts(cls) + length(cls_inds);
                    sums(cls, :) = sums(cls, :) + sum(bbox_targets(cls_inds, 2:end), 1);
                    squared_sums(cls, :) = squared_sums(cls, :) + sum(bbox_targets(cls_inds, 2:end).^2, 1);
                end
            end
        end
    end
end

bbox_means = bsxfun(@rdivide, sums, class_counts);
bbox_stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), bbox_means.^2)).^0.5;
% add background class
bbox_means = [0, 0, 0, 0; bbox_means];
bbox_stds = [0, 0, 0, 0; bbox_stds];
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