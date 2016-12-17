%% eval_rec_bb_video: evaluate recall at a fixed average spatial IOU threshold
function recall = eval_rec_bb_video(gt_rois, rois, threshold)
% gt_rois:       path to puppet mask (ground truth)
%                or gt structure array with field 'rois' and 'framespan'
% boxes_frames:  human_bb detection result num_frame*1 cells {N*5}
% recall:        the ratio of ground truth tubes covered by at least one
%                tube proposal with average spatial IoU over a theshold

% load bounding boxes
if ischar(gt_rois)
    if exist(gt_rois, 'file')
        % jhmdb gt is saved as matrix not struct
        ld = matfile(gt_rois, 'Writable', false);
        % jhmdb has whole framespan and one person (trimmed)
        bb_gt.framespan = [1 size(ld.BB, 2)];
        bb_gt.rois = ld.BB';
        clear ld;
    else
        fprintf(2, 'No puppet mask found at %s\n', gt_rois);
        return
    end
else
    bb_gt = gt_rois;
end
num_gt_persons = length(bb_gt);

% evaluate average spatial IoU over framespan
num_trupos = 0;
num_pos = num_gt_persons;
assert(num_gt_persons>0);
iou = 0;
for p = 1:num_gt_persons
    tube_length = bb_gt(p).framespan(2) - bb_gt(p).framespan(1) + 1;
    f_offset = bb_gt(p).framespan(1) - 1;
    for f = 1:tube_length
        % [x0_c, y0_c, x1_c, y1_c, prob_c]
        bb = rois{f + f_offset};
        % if no human detected
        if isempty(bb)
            iou = bsxfun(@plus, iou, 0);
            continue;
        end
        tmp = boxoverlap(bb_gt(p).rois(f,:), bb);
        % overlap with the maximum overlapping proposal
        iou = iou + max(tmp);
    end
    % average spatial IoU over framespan
    iou = single(iou)/single(tube_length);
    if iou > threshold
        num_trupos = num_trupos + 1;
    end
end
recall = num_trupos/num_pos;
end
