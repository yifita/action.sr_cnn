function [class_to_id, classes] = get_classID_map(imagenet_meta)
% GET_CLASSID_MAP helper function for retraining Faster-RCNN,
% create target classes from both imagenet and voc
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
VOC_classes={...
    'aeroplane'
    'bicycle'
    'boat'
    'bottle'
    'bus'
    'car'
%     'cat'
    'chair'
    'diningtable'
%     'dog'
    'horse'
    'motorbike'
    'person'
    'sofa'
    'train'
    'tvmonitor'
    };
VOC_synsets = {...
    'airplane'
    'bicycle'
    'watercraft'
    'bottle'
    'bus'
    'car'
%     'domestic cat'
    'chair'
    'table'
%     'dog'
    'horse'
    'motorcycle'
    'person'
    'sofa'
    'train'
    'tv or monitor'...
    };
imageNet_synsets = keys(imagenet_meta.ImageNet);
imageNet_wnids = values(imagenet_meta.ImageNet);
% renaming VOC classes to imagenet synsets
VOC_ImageNet_map = containers.Map(VOC_classes, VOC_synsets);
VOC_additional = setdiff(VOC_synsets, imageNet_synsets);
[~, common_idx, ~] = intersect(VOC_synsets, imageNet_synsets);

% get wnid of VOC classes
VOC_wnids = values(imagenet_meta.ImageNet, values(VOC_ImageNet_map, VOC_classes(common_idx)));
VOC_toplevel_wnids = values(imagenet_meta.children_parent_map, VOC_wnids);

% 150 wnids => 1:150
toplevel_id_map = containers.Map(values(imagenet_meta.ImageNet_toplevel), 1:length(imagenet_meta.ImageNet_toplevel));
% all wnids => 1:150 (children to parent)
imagenet_id_map = containers.Map(imageNet_wnids, values(toplevel_id_map, values(imagenet_meta.children_parent_map, imageNet_wnids)));
voc_to_id = containers.Map( VOC_classes(common_idx), values(imagenet_id_map, VOC_toplevel_wnids ));
class_to_id = [imagenet_id_map; voc_to_id ];
class_to_id = [class_to_id; containers.Map(VOC_additional, length(imagenet_meta.ImageNet_toplevel)+1:length(imagenet_meta.ImageNet_toplevel)+length(VOC_additional))];

classes = keys(imagenet_meta.ImageNet_toplevel);
classes = cat(1, classes(:), VOC_additional(:));
end
