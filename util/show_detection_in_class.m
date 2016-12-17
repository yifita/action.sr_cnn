function show_detection_in_class(dataset, data, clsName)
clsName_2_sample = containers.Map;
for i = 1:length(dataset.classes)
    clsName_2_sample(dataset.classes{i}) = find(strcmp(dataset.video_cls, dataset.classes{i}));
end
v_idx = clsName_2_sample(clsName);
for i = 1:length(v_idx)
    display_human_bb_video(dataset, v_idx(i), data(v_idx(i)).person_rois,'frame_skip',5);
    display_bb_video(dataset,v_idx(i),false,data(v_idx(i)).obj_rois);
end
end