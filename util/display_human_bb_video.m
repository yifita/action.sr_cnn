function display_human_bb_video(dataset, i, boxes_frames, varargin)
ip = inputParser;
ip.addParameter('gt_bb', false, @islogical);
ip.addParameter('frame_skip', 1, @isscalar);
ip.parse(varargin{:});
frame_skip = ip.Results.frame_skip;
% Display human_bb of ith video in dataset
%
% i             index
% show_gt       whether to display g

% given dataset and an index, display detected human bb in each frame
frames = dataset.frames_of(i);
if ~exist('boxes_frames', 'var') || isempty(boxes_frames)
    ld = load(dataset.feat_paths(dataset.faster_rcnn_dir, i), 'human_bb');
    % boxes N*(x0,y0,w,h)
    boxes_frames = ld.human_bb;
end
if strcmp(dataset.name, 'jhmdb')
    if length(boxes_frames) ~= dataset.num_frames(i)
        boxes_frames = boxes_frames(1:dataset.num_frames(i));
    end
else
    assert(length(boxes_frames) == dataset.num_frames(i), 'Number of frames mismatchin');
end
% load gt boxes (from puppet mask)
if ip.Results.gt_bb
    puppet = matfile(dataset.human_mask(i), 'Writable', false);
    % puppet.BB saved as 4xnum_frames
    gt_bb = puppet.BB';
    assert(size(gt_bb, 1) == dataset.num_frames(i), 'Number of frames mismatchin');
    gt_bb = num2cell(gt_bb, 2);
end
for j = 1:frame_skip:length(boxes_frames)
    im = imread(frames{j});
    imsize = size(im);
    if ~exist('gt_bb', 'var') || isempty(gt_bb)
        showboxes(im, boxes_frames(j), {'person'}, 'scale_rois', imsize(1:2)./dataset.frame_size(i,:));
    else
        showboxes(im, boxes_frames(j), {'person'}, 'gt_bb', gt_bb(j), 'scale_rois', imsize(1:2)./dataset.frame_size(i,:));
    end
    text(2,2, sprintf('%d/%d', j, dataset.num_frames(i)));
    waitforbuttonpress
end
