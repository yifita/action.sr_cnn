function puppetmask_to_bb(dataset, overwrite)
% GET_RAW_ROIS assemble a matlab structure containing detected rois
% for the whole dataset from image-wise Faster-RCNN output
% ---------------------------------------------------------
% Copyright (c) 2016, Yifan Wang
% 
% This file is part of the SR-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
if isempty(dataset)
    cache_file = 'imdb/cache/jhmdb_dataset.mat';
    ld = load(cache_file);
    dataset = ld.dataset;
end
if nargin < 2
    overwrite = true;
end
for i = 1:dataset.num_video
    tic_toc_print('%d/%d masks\n', i, dataset.num_video);

    try
        ld = matfile(dataset.human_mask(i),'Writable',true);
        if ~isempty(who(ld,'BB')) && ~overwrite
            continue;
        end
        puppet = ld.part_mask;

        % convert puppet mask to bounding boxes
        [rows, cols, ~] = find(puppet);
        [cols, frames] = ind2sub([size(puppet,2), size(puppet,3)], cols);

        num_frame = size(puppet, 3);
        assert(num_frame == dataset.num_frames(i), ...
            sprintf('%d puppet frames, %d video frames', num_frame, dataset.num_frames(i)));
        BB = nan(4, num_frame);
        for j = 1:num_frame
            % stats = regionprops(puppet(:,:,f), 'BoundingBox');
            y1 = max(rows(frames == j));
            x1 = max(cols(frames == j));
            y0 = min(rows(frames == j));
            x0 = min(cols(frames == j));
            BB(:, j) = [x0; y0; x1; y1];
        end
        ld.BB = BB;
    catch err
       disp(err.stack);
       fprintf('%s\n', err.message);
   end
end
