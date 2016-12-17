function recall = eval_recall_ucf(dataset, varargin)
% EVAL_RECALL_UCF evaluate person track on ucf dataset
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
ip.addRequired('dataset',              @isstruct);
ip.addOptional('rois',      [],        @isstruct);
ip.addOptional('rois_gt',   [],        @isstruct);
ip.addParameter('use_processed', true, @islogical);
ip.addParameter('threshold',     0.5,  @isscalar);
ip.addParameter('score_thres',   0.1,  @isscalar);
ip.addParameter('frame_interval',10,   @isscalar);
ip.addParameter('top_n_person',  2,    @isscalar);

ip.parse(dataset, varargin{:})
opts = ip.Results;

num_vids = sum(dataset.hasBB);
video_indices = find(dataset.hasBB);
recall = zeros(num_vids, 1);

if isempty(opts.rois)
    opts.rois = prepare_rois_context(dataset, [], 'train', opts);
end
if isempty(opts.rois_gt)
    ld = load('imdb/cache/ucf_BB_gt.mat');
    s = fieldnames(ld);
    assert(length(s)==1);
    opts.rois_gt = ld.(s{1});
    clear ld;
end
for i = 1:num_vids
    j = video_indices(i);
    recall(i) = eval_rec_bb_video(opts.rois_gt(j).person_rois, opts.rois(j).person_rois, opts.threshold);
end
end
