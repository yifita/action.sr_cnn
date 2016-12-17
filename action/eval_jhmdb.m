function recall = eval_jhmdb(dataset,varargin)
% evaluate person track
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addRequired('dataset',            @isstruct);
ip.addOptional('use_processed', {},  @islogical);
ip.addOptional('threshold',     {},  @isnumeric);
ip.parse(dataset, varargin{:})
opts = ip.Results;

os = cell(opts.dataset.num_video, 1);
recall = zeros(dataset.num_video, 2);
if ~opts.use_processed
    for j = 1:opts.dataset.num_video
        try
            bbfile = matfile(dataset.feat_paths(dataset.faster_rcnn_dir,j));
            boxes_frames = bbfile.human_bb;
            recall(j) = eval_video(dataset.human_mask(j), boxes_frames, opts.threshold);
        catch err
            fprintf('%s: %s\n', opts.dataset.video_ids{j}, err.message);
        end
    end
else
    opts.split = 2;
    opts.score_thres = 0.1;
    opts.frame_interval = 10;
    opts.top_n_person = 2;
    data = get_person_jhmdb(dataset, 1, 'train', opts);
    for j = 1:dataset.num_video
        recall(j,:) = eval_video(data(j).person_rois, boxes_frames, opts.threshold);
    end
end

%% eval_video: evaluate recall at a fixed average spatial IOU threshold
function recall = eval_video(puppet_mask, boxes_frames, threshold)
% puppet_mask       path to puppet mask (ground truth)
% boxes_frames      human_bb detection result num_frame*1 cells {N*5}
num_frames = length(boxes_frames);
num_tracks = size(boxes_frames{1}, 1);
iou = zeros(num_tracks,1);
% load bounding boxes
if exist(puppet_mask, 'file')
    ld = matfile(puppet_mask, 'Writable', false);
    bb_gt = ld.BB;
    clear ld;
else
    fprintf('No puppet mask found at %s\n', puppet_mask);
    return
end
% evaluate IOU per frame
for f = 1:num_frames
    % [x0_c, y0_c, x1_c, y1_c, prob_c]
    bb = boxes_frames{f};
    % if no human detected
    if isempty(bb)
        warning('Failed to detect humans in frame %d.', f);
        iou = bsxfun(@plus ,iou,  0);
        continue;
    end
    iou = iou + boxoverlap(bb, bb_gt(:,f)');
end
iou = iou/num_frame;
recall = double(iou > threshold);
recall(recall > 1) = 1;
fprintf('frame average AoI/AoC = %f\n', mean(o));
end
