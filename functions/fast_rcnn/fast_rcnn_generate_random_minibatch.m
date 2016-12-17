function [shuffled_inds, sub_inds, db_ind] = fast_rcnn_generate_random_minibatch(shuffled_inds, imdbs, ims_per_batch)
persistent hori_image_inds vert_image_inds db_prob
if isstruct(imdbs)
    imdbs = {imdbs};
end
if isempty(hori_image_inds)
    hori_image_inds = cell(length(imdbs),1);
    vert_image_inds = cell(length(imdbs),1);
    db_prob = cumsum(cellfun(@(x) length(x.image_ids), imdbs, 'UniformOutput', true));
    db_prob = db_prob/db_prob(end);
end
% pick one dataset w.r.t their size
p = rand(1); db_ind = 1;
for i = 1:length(imdbs)
    if p < db_prob(i)
        db_ind = i;
        break;
    else
        continue;
    end
end

% shuffle training data per batch
if isempty(shuffled_inds{db_ind})
    % make sure each minibatch, only has horizontal images or vertical
    % images, to save gpu memory
    if isempty(hori_image_inds{db_ind})
        hori_image_inds{db_ind} = imdbs{db_ind}.sizes(:,2)>imdbs{db_ind}.sizes(:,1);
        vert_image_inds{db_ind} = ~hori_image_inds{db_ind};
        hori_image_inds{db_ind} = find(hori_image_inds{db_ind});
        vert_image_inds{db_ind} = find(vert_image_inds{db_ind});
    end

    % random perm
    lim = floor(length(hori_image_inds{db_ind}) / ims_per_batch) * ims_per_batch;
    hori_image_inds{db_ind} = hori_image_inds{db_ind}(randperm(length(hori_image_inds{db_ind}), lim));
    lim = floor(length(vert_image_inds{db_ind}) / ims_per_batch) * ims_per_batch;
    vert_image_inds{db_ind} = vert_image_inds{db_ind}(randperm(length(vert_image_inds{db_ind}), lim));

    % combine sample for each ims_per_batch
    hori_image_inds{db_ind} = reshape(hori_image_inds{db_ind}, ims_per_batch, []);
    vert_image_inds{db_ind} = reshape(vert_image_inds{db_ind}, ims_per_batch, []);

    shuffled_inds{db_ind} = [hori_image_inds{db_ind}, vert_image_inds{db_ind}];
    shuffled_inds{db_ind} = shuffled_inds{db_ind}(:, randperm(size(shuffled_inds{db_ind}, 2)));

    shuffled_inds{db_ind} = num2cell(shuffled_inds{db_ind}, 1);
end

if nargout > 1
    % generate minibatch training data
    sub_inds = shuffled_inds{db_ind}{1};
    assert(length(sub_inds) == ims_per_batch);
    shuffled_inds{db_ind}(1) = [];
end
end