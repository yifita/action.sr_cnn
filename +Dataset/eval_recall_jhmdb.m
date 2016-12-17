function recall = eval_recall_jhmdb(dataset,varargin)
% EVAL_RECALL_JHMDB evaluates the person track for jhmdb
% dataset
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('dataset',            @isstruct);
ip.addOptional('rois',      [],        @isstruct);
ip.addOptional('rois_gt',   [],        @isstruct);
ip.addParameter('use_processed', true, @islogical);
ip.addParameter('threshold',     0.5,  @isscalar);
ip.addParameter('score_thres',   0.1,  @isscalar);
ip.addParameter('frame_interval',10,   @isscalar);
ip.addParameter('top_n_person',  2,    @isscalar);
ip.parse(dataset, varargin{:})
opts = ip.Results;

% load or detect and process person for jhmdb
if isempty(opts.rois)
    opts.rois = get_person_jhmdb(dataset, [], 'train', opts);
end
if isempty(opts.rois_gt)
    ld = load('imdb/cache/jhmdb_BB_gt.mat');
    s = fieldnames(ld);
    assert(length(s)==1);
    framespans = arrayfun(@(x) [1 length(x.person_rois)], ld.(s{1}), 'uni', false);
    rois = arrayfun(@(x) vertcat(x.person_rois{:}), ld.(s{1}), 'uni', false);
    opts.rois_gt = struct('framespan', framespans, 'rois', rois);
    clear ld;
end
recall = zeros(dataset.num_video, 1);
for j = 1:dataset.num_video
    recall(j) = eval_rec_bb_video(opts.rois_gt(j), opts.rois(j).person_rois, opts.threshold);
end
% end
end
