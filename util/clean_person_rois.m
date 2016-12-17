function data = clean_person_rois(dataset, data, classes)
function output = returnX(s, input)
    output = input;
end
for j = 1:length(classes)
    idx = find(strcmp(dataset.video_cls, classes{j}));
    for i = 1:length(idx)
        data(idx(i)).person_rois = cellfun(@(x) x(1,:), data(idx(i)).person_rois, 'uni',false, 'ErrorHandler', @returnX);
    end
end
end