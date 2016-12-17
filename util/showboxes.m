function showboxes(im, boxes, legends, varargin)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes)
%   append gt boxes
% -------------------------------------------------------
ip = inputParser;
ip.addParameter('gt_bb', {}, @iscell);
ip.addParameter('scale_rois', [1 1], @isnumeric);
ip.parse(varargin{:});
opts = ip.Results;

image(im);
axis image;
axis off;
set(gcf, 'Color', 'white');
if ~isempty(opts.gt_bb)
    assert(length(opts.gt_bb) == length(boxes));
    valid_boxes = cellfun(@(x, y) ~isempty(x)||~isempty(y), boxes, opts.gt_bb, 'UniformOutput', true);
else
    valid_boxes = cellfun(@(x) ~isempty(x), boxes, 'UniformOutput', true);
end
valid_boxes_num = sum(valid_boxes);

if valid_boxes_num > 0
    colors_candidate = colormap('hsv');
    colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/valid_boxes_num)):end, :);
    colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
    colors = cell(size(valid_boxes));
    colors(valid_boxes) = colors_candidate(1:sum(valid_boxes));
    
    for i = 1:length(boxes) % ith class
        for j = 1:size(boxes{i}) % jth object in class
            if size(boxes{i},2) > 5
                box = fast_rcnn_map_im_rois_to_feat_rois([], boxes{i}(j, 2:5), opts.scale_rois);
            else
                box = fast_rcnn_map_im_rois_to_feat_rois([], boxes{i}(j, 1:4), opts.scale_rois);
            end
            if size(boxes{i}, 2) >= 5
                score = boxes{i}(j, end);
                linewidth = 2 + min(max(score, 0), 1) * 2;
                rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{i});
                label = sprintf('%s : %.3f', legends{i}, score);
                text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
            else % draw ground truth with dashed lines
                linewidth = 2;
                rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{i});
                label = sprintf('%s(%d)', legends{i}, j);
                text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
            end
        end
    end
    for i = 1:length(opts.gt_bb)
        for j = 1:size(opts.gt_bb{i})
            box = opts.gt_bb{i}(j, 1:4) * opts.scale;
            linewidth = 2;
            rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{i}, 'LineStyle', '--');
            label = sprintf('%s(%d)', legends{i}, j);
            text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
        end
    end
end
end

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end
