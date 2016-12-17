function rm_roidb_from_proposal()
global dataset
for i = 1:length(dataset.roidb_train)
	for j = 1:length(dataset.roidb_train{i}.rois)
		gt_ind = dataset.roidb_train{i}.rois(j).gt;
		dataset.roidb_train{i}.rois(j).gt(~gt_ind) = [];
		dataset.roidb_train{i}.rois(j).boxes(~gt_ind,:) = [];
		dataset.roidb_train{i}.rois(j).class(~gt_ind) = [];
		dataset.roidb_train{i}.rois(j).overlap(~gt_ind,:) = [];
	end
end