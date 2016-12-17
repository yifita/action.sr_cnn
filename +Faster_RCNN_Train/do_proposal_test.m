function do_proposal_test(conf, model_stage, phase)
global dataset
if ~(strcmp('test',phase) || strcmp('train', phase))
    error('phase should be ''train'' or ''test''');
end
for db_idx = 1:length(dataset.(['imdb_' phase]))
    aboxes                      = proposal_test(conf, dataset.(['imdb_' phase]){db_idx}, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'rpn_keep_thres',   model_stage.rpn_keep_thres);

    boxes_filter(model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, ...
        conf.use_gpu);
    roidb_from_proposal(db_idx, aboxes, phase, ...
                                        'keep_raw_proposal', false);
    clear aboxes
end
    function boxes_filter(per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
        % to speed up nms
        if per_nms_topN > 0
            aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), aboxes, 'UniformOutput', false);
        end
        % do nms
        if nms_overlap_thres > 0 && nms_overlap_thres < 1
            if use_gpu
                for i = 1:length(aboxes)
                    if ~isempty(aboxes{i})
                        aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
                    end
                end
            else
                parfor i = 1:length(aboxes)
                    if ~isempty(aboxes{i})
                        aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
                    end
                end
            end
        end
        aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
        fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
        if after_nms_topN > 0
            aboxes = cellfun(@(x) x(1:min(length(x), after_nms_topN), :), aboxes, 'UniformOutput', false);
        end
    end
end
