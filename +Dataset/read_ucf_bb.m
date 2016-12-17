function person_gt = read_ucf_bb(dataset, xml_path)
%READ_UCF_BB reads human bounding box from thumos xml file
% and save in form of structure
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% Class Name mapping
% 'basketball_shooting' => 'Basketball'
% 'biking' => 'Biking'
% 'diving' => 'Diving'
% 'golf_swing' => 'GolfSwing'
% 'horse_riding' => 'HorseRiding'
% 'soccer_juggling' => 'SoccerJuggling'
% 'tennis_swing' => 'TennisSwing'
% 'trampoline_jumping' => 'TrampolineJumping'
% 'volleyball_spiking' => 'VolleyballSpiking'
% 'walking' => 'WalkingWithDog'
old_names = {
    'basketball_shooting'
    'biking'
    'diving'
    'golf_swing'
    'horse_riding'
    'soccer_juggling'
    'tennis_swing'
    'trampoline_jumping'
    'volleyball_spiking'
    'walking'
    };
new_names = {
    'Basketball'
    'Biking'
    'Diving'
    'GolfSwing'
    'HorseRiding'
    'SoccerJuggling'
    'TennisSwing'
    'TrampolineJumping'
    'VolleyballSpiking'
    'WalkingWithDog'
    };
naming_map = containers.Map(new_names, old_names);

[~, video_id, ~] = fileparts(xml_path);
video_idx = strcmp(dataset.video_ids, video_id);
video_cls = dataset.video_cls{video_idx};
num_frame = dataset.num_frames(video_idx);
person_gt = repmat(struct('rois', [0 0 0 0], 'framespan', [0 0]), 10, 1);

xml = xml2struct(xml_path);
if ~iscell(xml.viper.data.sourcefile)
    xml.viper.data.sourcefile = {xml.viper.data.sourcefile};
end
% only the first roucefile is the new ucf101 video
objects = xml.viper.data.sourcefile{1}.object;
if ~iscell(objects)
    objects = {objects};
end

is_actor = false(10,1);
p_counter = 0;
for p = 1:length(objects)
    if ~strcmp(objects{p}.Attributes.name, 'PERSON')
        continue;
    end
    % don't record human bb if this person doesn't involve in the action
    if ~exist('attr_idx', 'var')
        cls_names = cellfun(@(x) x.Attributes.name, objects{p}.attribute, 'uni',false);
        old_anno = length(objects{p}.attribute) ~= (length(dataset.classes) + 1);
        if old_anno
            video_cls = naming_map(video_cls);
        end
        attr_idx = strcmp(cls_names, video_cls);
    end
    assert(strcmp(objects{p}.attribute{attr_idx}.Attributes.name, video_cls));
    if isstruct(objects{p}.attribute{attr_idx}.data_colon_bvalue)
        is_actor(p) = parseLogical(objects{p}.attribute{attr_idx}.data_colon_bvalue.Attributes.value);
    else
        is_actor(p) = any(cellfun(@(x) parseLogical(x.Attributes.value), objects{p}.attribute{attr_idx}.data_colon_bvalue));
    end
    p_counter = p_counter + 1;

    %% extract person tubes
    % framespan
    framespan = parseTimeSpan(objects{p}.Attributes.framespan);
    if framespan(2) > num_frame
        framespan(2) = num_frame;
    end
    person_gt(p_counter).framespan = framespan;
    tube_length = framespan(2)-framespan(1)+1;

    % rois
    person_gt(p_counter).rois = zeros(tube_length, 4);
    locations = objects{p}.attribute{1};
    assert(strcmp(locations.Attributes.name, 'Location'));
    for t = 1:length(locations.data_colon_bbox)
        t_tmp = parseTimeSpan(locations.data_colon_bbox{t}.Attributes.framespan);
        %
        if t_tmp(2) < framespan(1)
            continue;
        end
        if t_tmp(1) > framespan(2)
            continue;
        end
        % ground truth as zero indexing
        l_tmp = [str2double(locations.data_colon_bbox{t}.Attributes.x)+1,...
            str2double(locations.data_colon_bbox{t}.Attributes.y)+1,...
            str2double(locations.data_colon_bbox{t}.Attributes.width),...
            str2double(locations.data_colon_bbox{t}.Attributes.height)];
        l_tmp(3:4) = l_tmp(1:2) + l_tmp(3:4);
        % gt includes bb that are outside the image boundary
        l_tmp = fixBoundary(l_tmp, dataset.frame_size(video_idx, 2), dataset.frame_size(video_idx, 1));
        if isempty(l_tmp)
            continue;
        end
        if t_tmp(2) > framespan(2)
            t_tmp(2) = framespan(2);
        end
        if t_tmp(1) < framespan(1)
            t_tmp(1) = framespan(1);
        end
        t_offset = -framespan(1) + 1;
        person_gt(p_counter).rois(t_tmp(1)+t_offset:t_tmp(2)+t_offset,:) = repmat(l_tmp, t_tmp(2)-t_tmp(1)+1,1);
        if t_tmp(2) >= framespan(2)
            break;
        end
    end
end
is_actor(p_counter+1:end) = [];
person_gt(p_counter+1:end) = [];
% some gt xml doesn't mark any person as actor, in this case treat all
% persons as actor
if ~all(is_actor) && any(is_actor)
    person_gt(~is_actor) = [];
end
end
%% parseLogical: function description
function flag = parseLogical(str)
true_strs = {'true', '1', 'True', 'TRUE'};
flag = any(strcmp(str, true_strs));
end
%% parseTimeSpan: parse 'x:y' string to t1 and t2
function span = parseTimeSpan(arg)
    tmp = textscan(arg, '%d:%d');
    span = [tmp{1}, tmp{2}];
end
%% fixBoundary: function description
function bb = fixBoundary(bb, width, height)
    if (bb(1) < 1 && bb(3) <= 1) || (bb(2) < 1 && bb(4) <= 1) ||...
            (bb(1) >= width && bb(3) > width) || (bb(2) > height && bb(4) > height)
        bb = [];
        return;
    end
	if bb(1) < 1
		bb(1) = 1;
	end
	if bb(2) < 1
		bb(2) = 1;
	end
	if bb(3) > width
		bb(3) = width;
	end
	if bb(4) > height
		bb(4) = height;
	end
end