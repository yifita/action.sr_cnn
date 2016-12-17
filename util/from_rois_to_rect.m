function rect = from_rois_to_rect(rois)
rect = [rois(:,[1 2]) rois(:,[3 4]) - rois(:,[1 2])];