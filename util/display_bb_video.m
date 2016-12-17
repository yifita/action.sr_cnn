function display_bb_video(dataset, i, boxes_frames, varargin)
ip = inputParser;
ip.addParameter('sub_class', {} , @iscell);
ip.addParameter('score_thres', 0.1, @isscalar);
ip.parse(varargin{:});
opts = ip.Results;
% Display human_bb of ith video in dataset
%
% i             index
% show_gt       whether to display g
opts.isVOC
imagenet_meta = Dataset.script_get_wnid();
[cls_to_id, classes] = Dataset.get_classID_map(imagenet_meta); 

sub_ids = 1:length(classes);
if ~isempty(opts.sub_class)
    if ~iscell(opts.sub_class)
        opts.sub_class = {opts.sub_class};
    end
    sub_ids = cell2mat(cls_to_id.values(opts.sub_class));
end
% given dataset and an index, display detected human bb in each frame
frames = dataset.frames_of(i);
if ~exist('boxes_frames', 'var') || isempty(boxes_frames)
    ld = load(dataset.feat_paths(dataset.faster_rcnn_dir, i), 'boxes_frames');
%     boxes_frames = ld.boxes_frames(2:end);
    boxes_frames = ld.boxes_frames;
end
% boxes N*(x0,y0,w,h)
assert(length(boxes_frames) == dataset.num_frames(i), 'Number of frames mismatchin');

for j = 1:dataset.num_frames(i)
    im = imread(frames{j});
    imsize = size(im);
    if size(boxes_frames{j},2)<=5 || iscell(boxes_frames{j})
        tmp = cellfun(@(x) x(x(:,end)>opts.score_thres,:), boxes_frames{j}, 'uni',false);
        showboxes(im, tmp(sub_ids), classes(sub_ids), 'scale_rois', imsize(1:2)./dataset.frame_size(i,:));
    else
        if size(boxes_frames{j},1) > 0
            boxes_frames{j} = boxes_frames{j}(boxes_frames{j}(:,end)>=opts.score_thres, :);
        end
        myshowboxes(im, boxes_frames{j}, classes);
    end
    text(2,2,sprintf('%d/%d', j, dataset.num_frames(i)))
    waitforbuttonpress
end
end
function myshowboxes(im, boxes, legends, varargin)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes)
%   append gt boxes
% -------------------------------------------------------
ip = inputParser;
ip.addParameter('gt_bb', {}, @iscell);
ip.parse(varargin{:});
opts = ip.Results;

fix_width = 320;
if isa(im, 'gpuArray')
    im = gather(im);
end
imsz = size(im);
scale = fix_width / imsz(2);
im = imresize(im, scale);

image(im);
axis image;
axis off;
set(gcf, 'Color', 'white');

% colors = colormap('hsv');
% colors = colors(1:(floor(size(colors, 1)/length(legends))):end, :);
% colors = mat2cell(colors, ones(size(colors, 1), 1))';

for i = 1:length(legends) % ith class
    boxes_c = boxes(boxes(:,1) == i, :);
    for j = 1:size(boxes_c,1)
        box = boxes_c(j,2:5) * scale;
        score = boxes_c(j,6);
        linewidth = 1 + min(max(score, 0), 1) * 2;
        rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', 'r');
        label = sprintf('%s : %.3f', legends{i}, score);
        text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');

    end
end
for i = 1:length(opts.gt_bb)
    for j = 1:size(opts.gt_bb{i})
        box = opts.gt_bb{i}(j, 1:4) * scale;
        linewidth = 2;
        rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{i}, 'LineStyle', '--');
        label = sprintf('%s(%d)', legends{i}, j);
        text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
    end
end

end
function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end