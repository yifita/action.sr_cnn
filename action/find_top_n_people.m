function [person_boxes_frames] = find_top_n_people(boxes_frames, x_flow_paths, y_flow_paths,...
    class_id, frame_interval, top_n, score_thres, img_width, img_height)
% FIND_TOP_N_PEOPLE finds the top-n relevant person tracks in a video
% requires extracted optical flow
% - boxes_frames: cell array saved from faster_rcnn_{dataset}
% - x_flow_paths: cell array containing the flow image
% - y_flow_paths: cell array containing the flow image
% - class_id: class idx (1-based) for person
% - frame_interval: number of frame_interval frames will be considered
%   as one shortest-path node
% - top_n: find the top-n tracks
% - score_thres: tracks with score below this will be discarded
% - img_width: frame width
% - img_height: frame height
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

human_boxes = cellfun(@(frame) frame{class_id}, boxes_frames, 'uni', false);
num_frame = length(human_boxes);
person_boxes_frames = cell(size(human_boxes));
[person_boxes_frames{:}] = deal(zeros(top_n, 4, 'single'));

% filter boxes with threshold < score_thres
if exist('score_thres', 'var')
    for k = 1:num_frame
        if ~isempty(human_boxes{k})
            human_boxes{k}(human_boxes{k}(:,5) < score_thres,:) = [];
        end
    end
end
tmp = imfinfo(x_flow_paths{1});
assert(isequal([tmp.Height tmp.Width], [img_height, img_width]));
for i = 1:top_n
    % find first and last frame with a successful object detection
    start_frame = num_frame;
    end_frame = 1;
    for k = 1:num_frame
        if ~isempty(human_boxes{k})
            if start_frame > k
                start_frame = k;
            end
            if end_frame <= k
                end_frame = k;
            end
        end
    end
    intervals = start_frame:frame_interval:end_frame;
    if numel(intervals) <= 2
        break;
    end
    % group n=frame_interval frames
    nodes = arrayfun(@(i) ... //starting frame of an interval 1, 6, ...
        cell2mat(...
        cellfun(@(f, x) ... // (frame index, boxes)
        [repmat(f, [size(x,1),1]) x], ...
        num2cell([i:min(i+frame_interval-1,end_frame)]',2), human_boxes(i:min(i+frame_interval-1,end_frame)),'uni',false)), ...
        intervals','uni',false);
    
    % stop searching if more than a half intervals are empty
    empty_idx = cellfun(@isempty, nodes, 'uni',true);
    nodes(empty_idx) = [];
    if length(nodes) <= 0.5*length(intervals)
        break;
    end
    intervals(empty_idx) = [];
    % dynamic programming. get optimal tube
    [J, U] = dynamic_prog_score_action(nodes, x_flow_paths, y_flow_paths); % J maximizing score reaching current node, U optimal strategy reaching current node
    if max(J{1})<0
        break;
    end
    % backtrace U to get choice of person bb in each frame
    boxes = zeros(length(nodes), 4,'single');
    frame_ind = zeros(length(nodes), 1, 'single');
    for k = 0:length(nodes)-1
        if k == 0
            [~, p] = max(J{1});
        else
            p = U{k}(p);
        end
        % add selected bbox
        boxes(k+1,:) = nodes{k+1}(p,2:5);
        frame_ind(k+1) = nodes{k+1}(p,1);
    end
    tubes = zeros(end_frame - start_frame + 1, 4, 'single');
    tubes(frame_ind,:) = boxes;
    
    % interpolate and extrapolate
    missing_idx = setdiff(1:num_frame, frame_ind);
    ex_interpolated = ...
        interp1(frame_ind,boxes,missing_idx, 'linear', 'extrap');
    tubes(missing_idx,:) = ex_interpolated;
    
    % error caused by extrapolation
    tubes = fix_exinterpolation_outofbound(tubes, frame_ind, img_width, img_height);
    
    for k = 1:num_frame
        % remove bb of this route from boxes_frames
        max_bb_ind = find(boxoverlap(human_boxes{k},tubes(k,:)) > 0.64);
        for j = 1:numel(max_bb_ind)
            % don't remove bb if multiple bb from the same frame
            % links to it
            key_node = false;
            if k > intervals(2)
                interval = find(intervals<=k, 1, 'last');
                offset = find(nodes{interval}(:,1) == k, 1);
                coming_nodes = U{interval-1} == (max_bb_ind(j) + offset - 1);
                frame_coming_nodes = nodes{interval-1}(coming_nodes,1);
                if any(diff(frame_coming_nodes)==0)
                    key_node = true;
                end
            end
            if ~key_node
                human_boxes{k}(max_bb_ind(j),:) = [];
            end
        end
        person_boxes_frames{k}(i,:) = tubes(k,:);
    end
end
% per_frame_score = per_frame_score(1:counter);
% purge zero cells
for k = 1:num_frame
    person_boxes_frames{k}(~all(person_boxes_frames{k}, 2),:) = [];
end
end

%% fix out of boundary mistakes due to extrapolation
function tubes = fix_exinterpolation_outofbound(tubes, frame_ind, img_width, img_height)
tubes(tubes(:, 1)<1, 1) = 1;
tubes(tubes(:, 2)<1, 2) = 1;
tubes(tubes(:, 3)>img_width, 3) = img_width;
tubes(tubes(:, 4)>img_height, 4) = img_height;
invalid_w = find(tubes(:,3)<=tubes(:,1));
invalid_h = find(tubes(:,4)<=tubes(:,2));
for j = 1:numel(invalid_w)
    % find the closest frame found by dynamic programming (this could
    % either be the first or last)
    [~,f] = min(abs(frame_ind - invalid_w(j)));
    f = frame_ind(f);
    width = tubes(f,3) - tubes(f,1)+1;
    if tubes(invalid_w(j), 1) > 1 && tubes(invalid_w(j), 1) < img_width
        tubes(invalid_w(j), 3) = min(img_width, 1.5*width + tubes(invalid_w(j), 1) -1);
    elseif tubes(invalid_w(j), 3) < img_width && tubes(invalid_w(j), 3) > 1
        tubes(invalid_w(j), 1) = max(1, -1.5*width + tubes(invalid_w(j), 3)+1);
    else
        tubes(invalid_w(j), [1 3]) = [1 img_width];
    end
end
for j = 1:numel(invalid_h)
    % find the closest frame found by dynamic programming (this could
    % either be the first or last)
    [~, f] = min(abs(frame_ind - invalid_h(j)));
    f = frame_ind(f);
    height = tubes(f,4) - tubes(f,2)+1;
    if tubes(invalid_h(j), 2) > 1 && tubes(invalid_h(j), 2) < img_height
        tubes(invalid_h(j), 4) = min(img_height, 1.2*height + tubes(invalid_h(j), 2) -1);
    elseif tubes(invalid_h(j), 4) < img_height && tubes(invalid_h(j), 4) > 1
        tubes(invalid_h(j), 2) = max(1, -1.2*height + tubes(invalid_h(j), 4)+1);
    else
        tubes(invalid_h(j), [2 4]) = [1 img_height];
    end
end
end

%% Dynamic programming find maximizing path
function [J,U] = dynamic_prog_score_action(nodes, x_flow_paths, y_flow_paths)
J = cell(length(nodes), 1);
U = cell(length(nodes), 1);
flow_m = cellfun(@(x, y) normalized_flow(x,y), x_flow_paths, y_flow_paths, 'uni', false, 'ErrorHandler', @returnEmpty);
for k = length(nodes):-1:1
    % maximize over i
    % J{k} j*1
    % J{k+1}	 i*1
    if k < length(nodes)
        Cost2Go = bsxfun(@plus, immediate_score(nodes, k), J{k+1}');
    else
        Cost2Go = immediate_score(nodes, k);
    end
    % J{k} cost to go from current node
    % U{k} optimal strategy from current node
    [J{k}, U{k}] = max(Cost2Go, [], 2);
    U{k}(J{k} == -inf) = nan;
end
%% Score to maximize prob + AoI/AoU + areascore
    function cost = immediate_score(nodes, k)
        % boxoverlap num_j*num_i
        areas = cellfun(@(x) area(x(2:5)), num2cell(nodes{k},2));
        % non-linear area score f(a), strong penalty on small areas
        map_fun = @(x) 0.5/(1+exp(-12*(x-0.5)));
        area_score = arrayfun(map_fun, areas/max(areas), 'uni',true);
        % if there's no flow, then set default flow_score to be 1
        flow_score = cellfun(@(x) flow_score(flow_m{x(1)}, x(2:5)), num2cell(nodes{k},2),'ErrorHandler',@returnNaN);
        flow_score = arrayfun(map_fun, flow_score/(max(flow_score)+eps),'uni',true);
        flow_score(isnan(flow_score)) = 1;
        if k < length(nodes)
            overlap_score = boxoverlap(nodes{k}(:, 2:5), nodes{k+1}(: ,2:5));
            overlap_score(overlap_score == 0) = -20;
            cost = repmat(nodes{k}(:, end)+area_score+2*flow_score, [1, size(nodes{k+1},1)]) ...
                + overlap_score;
        else
            cost = nodes{k}(:, end) + area_score+flow_score;
        end
    end

end

%% Area of bb
function a=area(b)
a = (b(3)-b(1)+1) * (b(4)-b(2)+1);
end
%% flow score
function flow = normalized_flow(x_flow_path, y_flow_path)
x_flow = single(imread(x_flow_path));
x_flow = (x_flow - mean(x_flow(:)));
y_flow = single(imread(y_flow_path));
y_flow = y_flow - mean(y_flow(:));
flow = sqrt(x_flow.*x_flow+ y_flow.*y_flow);
flow = flow/max(flow(:));
end
function score = flow_score(flow, rois)
rois = round(rois);
flow = flow(rois(2):rois(4),rois(1):rois(3));
score = sum(flow(:))/area(rois);
end

%% Error Handler
function output = returnEmpty(s, varargin)
output = [];
end
function output = returnNaN(s, varargin)
output = nan('single');
end
