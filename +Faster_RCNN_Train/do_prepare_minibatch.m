function image_roidb = do_prepare_minibatch(conf, imdb, roidb, sub_db_inds, bbox_means, bbox_stds)
% Gather useful information from imdb and roidb
%   pre-calculate mean (bbox_means) and std (bbox_stds) of the regression
%   term for normalization

image_roidb = ...
    arrayfun(@(z) ... //@([1:length(imdb.image_ids)])
                struct('image_path', imdb.image_at(z), 'image_id', ...
                    imdb.image_ids{z}, 'im_size', imdb.sizes(z, :), 'imdb_name', imdb.name, ...
                    'overlap', roidb.rois(z).overlap, 'boxes', roidb.rois(z).boxes, ...
                    'class', roidb.rois(z).class, 'image', [], 'bbox_targets', []), ...
            sub_db_inds, 'UniformOutput', true);

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
            fprintf('Warning: do_prepare_minibatch: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
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