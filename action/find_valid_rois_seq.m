function [person_boxes_frames, object_boxes_frames] = find_valid_rois_seq(dataset, v_idx, varargin)
% FIND_VALID_ROIS_SEQ filter out outlier detection boxes by computing a score for given sequence
% For human bounding box, select a tube by maximizing score function and
% (optionally) interpolate missing bb.
% For other objects, filter sporatically appearing bb by checking
% consistency across frames.
% Args:
% - dataset: output from get_dataset_{datasetname}
% - v_idx: video idx (1-based)
% Output:
% - boxes_frames {C {N X 5}} after nms
% - person_boxes_frames {{N X 5} X frames}
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
% img size (for exterpolation
ip.addParameter('img_width', 320, @isscalar);
ip.addParameter('img_height', 240, @isscalar);
% maximal number of person bb to be found
ip.addParameter('top_n_person', 2, @isscalar);
% threshold for proposal score
ip.addParameter('score_thres',0.1, @isscalar);
% nms threshold to filter out objects detection that overlaps with person
% detection
ip.addParameter('nms_thres', 0.7, @isscalar);
ip.addParameter('frame_interval', 10, @isscalar);
ip.parse(varargin{:});
opts = ip.Results;
%% 
ld = load('imagenet_meta');
imagenet_meta = ld.imagenet_meta;
[~, classes] = Dataset.get_classID_map(imagenet_meta);
%% 
%  load boxes_frames
ld = load(dataset.feat_paths(dataset.faster_rcnn_dir, v_idx));
s = fieldnames(ld);
assert(length(s)==1);
boxes_frames = ld.(s{1});
%  hmdb dataset has one rgb image too many 0000.jpg 
if strcmp(dataset.name, 'hmdb') && dataset.num_frames(v_idx) == (length(boxes_frames)-1)
    boxes_frames(1) = [];
end
    
person_class_id = find(strcmp(classes,'person'));
% person_class_id = 68;
%% Dynamic programming search from existing human detections
% initialize choice person bounding boxes
flow_x_paths = dataset.flow_x_paths(v_idx);
flow_y_paths = dataset.flow_y_paths(v_idx);
person_boxes_frames = find_top_n_people(boxes_frames, flow_x_paths, flow_y_paths,...
    person_class_id, opts.frame_interval, opts.top_n_person, opts.score_thres, opts.img_width, opts.img_height);

%% Other detected objects
% NumFrame cells containing [class_id x1 y1 x2 y2 cls_score]
object_boxes_frames = cellfun(@(x, y) filter_boxes(x, y, person_class_id, opts.score_thres, opts.nms_thres, opts.img_width, opts.img_height),...
    boxes_frames, person_boxes_frames, 'uni',false);

end

function object_boxes = filter_boxes(frame_boxes, detected_person, person_class_id, score_thres, nms_thres, width, height)
% FORBIDDEN_BORDER = 5;
obj_cls_idx = [1:person_class_id-1, person_class_id+1:length(frame_boxes)];
obj_cls_idx_cell = num2cell(obj_cls_idx,1)';
% nms with detected person
% filter score<threshold, prepend class_idx information
object_boxes = cellfun(@(obj_class, cls_idx) ...//class
        [repmat(cls_idx, sum(obj_class(:,end)>score_thres), 1) obj_class(obj_class(:,end)>score_thres,:)], ...
        frame_boxes(obj_cls_idx), obj_cls_idx_cell, 'uni',false);
object_boxes = cat(1, object_boxes{:});
% we don't want bounding boxes next to border
% invalid = (object_boxes(:,2) < FORBIDDEN_BORDER | object_boxes(:,3) < FORBIDDEN_BORDER | object_boxes(:,4) > width-FORBIDDEN_BORDER | object_boxes(:,5) > height-FORBIDDEN_BORDER);
% object_boxes(invalid,:) = [];
if isempty(detected_person)
    return
end
if size(detected_person,2) < 5
    detected_person = [detected_person 0.9*ones(size(detected_person,1),1)];
end
all_boxes = cat(1, object_boxes(:,2:end), detected_person);
after_nms = nms(all_boxes, nms_thres);
after_nms = setdiff(after_nms, size(object_boxes,1)+(1:size(detected_person,1)), 'stable');
object_boxes = object_boxes(after_nms,:);
detected_person = detected_person(:,1:4);
person_center = [detected_person(:,1) + detected_person(:,3) detected_person(:,2) + detected_person(:,4)]/2;
% in the 2X vincinity of the detected person, calculate object overlapping
person_region = 1.2*(detected_person - person_center(:,[1 2 1 2]))+person_center(:,[1 2 1 2]);
person_region(person_region(:,1)<1,1) = 1;
person_region(person_region(:,2)<1,2) = 1;
person_region(person_region(:,3)>width,3) = width;
person_region(person_region(:,4)>height,4) = height;
inter = boxinter(object_boxes(:,2:5), person_region, 0);
% only keep those that have interception with person
object_boxes = object_boxes( any(inter>0, 2), :);
end
