function confusion = list_confusion_class(dataset, confusion_matrix, class_name)
cls_2_id = containers.Map(dataset.classes, 1:length(dataset.classes));
if iscell(confusion_matrix)
    confusion = cellfun(@(x) get_confusions(x), confusion_matrix, 'uni',false);
else
    confusion = get_confusions(confusion_matrix);
end
    function confusion = get_confusions(confusion_matrix)
        [prob, id] = sort(confusion_matrix(cls_2_id(class_name),:),'descend');
        tmp = prob == 0 | id == cls_2_id(class_name);
        id(tmp) = [];
        prob(tmp) = [];
        confusion = struct('class',dataset.classes(id),'prob',num2cell(reshape(prob,[],1),2));
    end
end