function write_data_txt(dataset, data)
% create list file for video data, which is used by limin caffe
% VideoDataLayer

shuffled_idx = 1:dataset.num_video;
fid1 = fopen(sprintf('imdb/cache/%s_video_list.txt', dataset.name),'w');
fid2 = fopen(sprintf('imdb/cache/%s_roi_list.txt', dataset.name),'w');
cls_id = containers.Map(dataset.classes, 0:length(dataset.classes)-1);
for v=1:dataset.num_video
    i = shuffled_idx(v);
    fprintf(fid1, '%s/%s/%s %d %d\n', dataset.root, dataset.video_cls{i}, dataset.video_ids{i}, ...
        dataset.num_frames(i), cls_id(dataset.video_cls{i}));
    x_scale = 340/dataset.frame_size(i, 2);
    y_scale = 256/dataset.frame_size(i, 1);
    for f = 1:dataset.num_frames(i)
        if size(data(i).person_rois{f}, 1) < 1
            rois_str = '-';
        else
            person_rois = round(bsxfun(@times, data(i).person_rois{f}-1, [x_scale, y_scale, x_scale, y_scale]));
            rois_str = sprintf('%d,', person_rois);
            rois_str = rois_str(1:end-1);
        end
        fprintf(fid2, '%d %d %s ', v-1, f-1, rois_str);
        if size(data(i).obj_rois{f} ,1) < 1
            rois_str = '-';
        else
            tmp = data(i).obj_rois{f}(:,2:5);
            obj_rois = round(bsxfun(@times, tmp-1, [x_scale, y_scale, x_scale, y_scale]));
            rois_str = sprintf('%d,', obj_rois);
            rois_str = rois_str(1:end-1);
        end
        fprintf(fid2, '%s\n', rois_str);
    end
end