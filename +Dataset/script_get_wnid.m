function output = script_get_wnid(root_dir)
% SCRIPT_GET_WNID maps imagenet synsets to human readable synsets, 
% handles child synsets
% Arg:
% - root_dir: directory where meta_det.mat is saved
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if nargin == 0
    root_dir = fileparts(mfilename('fullpath'));
end
output = struct();

%------------------ Manual Config ---------------------%
ImageNet_Exclusion = {...
    'ant'
    'antelope'
    'armadillo'
    'artichoke'
    'apple'
    'bagel'
    'banana'
    'banjo'
    'bear'
    'bee'
    'bell pepper'
    'binder'
    'bird'
    'bow tie'
    'burrito'
    'butterfly'
    'camel'
    'cattle'
    'centipede'
    'chime'
    'cream'
    'crutch'
    'cucumber'
    'dish washer'
    'dog'
    'domestic cat'
    'dragonfly'
    'elephant'
    'fig'
    'filing cabinet'
    'flute'
    'french horn'
    'fox'
    'frog'
    'giant panda'
    'goldfish'
    'guacamole'
    'hamster'
    'head cabbage'
    'hippopotamus'
    'hotdog'
    'isopod'
    'jellyfish'
    'koala bear'
    'ladybug'
    'lemon'
    'lion'
    'lizard'
    'lobster'
    'maraca'
    'monkey'
    'oboe'
    'otter'
    'orange'
    'pencil sharpener'
    'pencil box'
    'perfume'
    'plate rack'
    'pitcher'
    'porcupine'
    'rabbit'
    'ray'
    'red panda'
    'scorpion'
    'seal'
    'sheep'
    'skunk'
    'snail'
    'snake'
    'spatula'
    'strainer'
    'strawberry'
    'starfish'
    'Stethoscope'
    'stretcher'
    'squirrel'
    'syringe'
    'swine'
    'tape player'
    'tiger'
    'tick'
    'turtle'
    'waffle iron'
    'whale'
    'zebra'
    };
%% Overlapping with ILSVRC2014 Detection set
tmp = load(fullfile(root_dir, 'meta_det.mat'), 'synsets');
% exclude manually selected categories
[toplevel_label, idx] = setdiff({tmp.synsets(1:200).name}, ImageNet_Exclusion);
children_idx = get_children(tmp.synsets, idx);
ImageNet_wnid = {tmp.synsets(children_idx).WNID}';
ImageNet_label = {tmp.synsets(children_idx).name}';
output.ImageNet_toplevel = containers.Map(toplevel_label, {tmp.synsets(idx).WNID}');
output.ImageNet = containers.Map(ImageNet_label,ImageNet_wnid);
output.toplevel_idx = idx;
% build a child-parent map that maps sub-categories to top categories
[~, idx] = setdiff({tmp.synsets(1:200).name}, ImageNet_Exclusion);
map = containers.Map;
for l = 1:length(idx)
    children_idx = get_children(tmp.synsets, idx(l));
    map = [map ; containers.Map( {tmp.synsets(children_idx).WNID}, repmat({tmp.synsets(idx(l)).WNID},[length(children_idx), 1]))];
end
output.children_parent_map = map;
end

function visited = get_children(synsets, idx)
pending = idx;
visited = [];
while ~isempty(pending)
    i = pending(1);
    recursive_depth(i);
end
    function recursive_depth(i)
        curr_children = synsets(i).children;
        tbd_children = setdiff(curr_children, visited);
        if isempty(tbd_children)
            pending = pending(2:end);
            visited = vertcat(visited, i);
        else
            pending = vertcat(tbd_children', pending);
        end
        return;
    end
end