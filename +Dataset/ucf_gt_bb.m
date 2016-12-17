function [dataset, gt_bb] = ucf_gt_bb(dataset, anno_dir)
% UCF_GT_BB assemble a matlab structure array containing the 
% ground truth person bounding boxes for UCF dataset from ucf
% annotations
anno_classes = {
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
gt_bb = repmat(struct('person_rois',[]), dataset.num_video, 1);
dataset.hasBB = false(dataset.num_video, 1);
for j = 1:length(anno_classes)
    video_idx = find(strcmp(dataset.video_cls, anno_classes{j}));
    for i = 1:length(video_idx)
        try
            xml_path = sprintf('%s/%s/%s.xgtf', ...
                anno_dir, dataset.video_cls{video_idx(i)}, dataset.video_ids{video_idx(i)});
            if exist(xml_path, 'file')
                gt_bb(video_idx(i)).person_rois = Dataset.read_ucf_bb(dataset, xml_path);
                dataset.hasBB(video_idx(i)) = true;
            else
                warning('Cound''nt find %s!', xml_path);
            end
        catch e
            fprintf(2, '%s(%d):%s\n', e.stack(1).file, e.stack(1).line, e.message);
        end
    end
end