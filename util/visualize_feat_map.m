function visualize_feat_map(net_inputs, feat, feat_bs)
close all;
f0 = figure('Name', 'Input', 'NumberTitle', 'off');
% title('input');
screensize = get( groot, 'Screensize');
f1 = figure('Name', 'conv5_3 p+s', 'NumberTitle', 'off', 'Position',[screensize(3:4)/2-1 screensize(3:4)/2]);
% title('conv5_3 ours', 'Interpreter', 'none');
f2 = figure('Name', 'conv5_3 scene', 'NumberTitle', 'off', 'Position',[0 0 screensize(3:4)/2]);
% title('conv5_3 theirs', 'Interpreter', 'none');
% rows_per_fig = 4;
for i = 1:length(net_inputs)
    % rgb input
    im_blob = net_inputs{i}{1};
    figure(f0);
    imshow(permute(uint8(im_blob(:,:,[3 2 1],i)+128),[2 1 3]),'Border','tight');
    person_blob = net_inputs{i}{2};
    person_blob = permute(person_blob, [4 3 2 1]);
    person_rois = person_blob(1,2:5);
    rectangle('Position', from_rois_to_rect(person_rois+1),'EdgeColor','r');
    % featmap
%     figure(f1)
    featmap = permute(feat{i}, [3 4 2 1]);
    featmap = rescale(featmap(:,:,:,1));
    featmap_bs = permute(feat_bs{i}, [3 4 2 1]);
    featmap_bs = rescale(featmap_bs(:,:,:,1));

    % randomly select 64 feature maps
    num_channels = size(featmap, 3);
    ch_idx = randperm(num_channels, 16*16);
    featmap = featmap(:,:,ch_idx,1);
    featmap_bs = featmap_bs(:,:,ch_idx,1);
%     for k = 0:rows_per_fig:size(featmap, 3)-1
%         for j = 0:rows_per_fig-1
%             subplot(4, 2, j*2+1)
%             imshow(featmap(:,:,k+j+1));
%             rectangle('Position', from_rois_to_rect(person_rois/16),'EdgeColor','r');
%             title('ours')
%             subplot(4, 2, j*2+2)
%             imshow(featmap_bs(:,:,k+j+1));
%             rectangle('Position', from_rois_to_rect(person_rois/16),'EdgeColor','r');
%             title('baseline')
%         end
%         waitforbuttonpress
%     end
    figure(f1)
    ha = tight_subplot(16,16,[0.01 0.01],0,0);
    for k = 1:length(ch_idx)
        axes(ha(k));
        imshow(featmap(:,:,k));
        rectangle('Position', from_rois_to_rect(person_rois/16),'EdgeColor','r');
    end
    figure(f2)
    ha_bs = tight_subplot(16,16,[0.01 0.01],0,0);
    for k = 1:length(ch_idx)
        axes(ha_bs(k));
        imshow(featmap_bs(:,:,k));
        rectangle('Position', from_rois_to_rect(person_rois/16),'EdgeColor','r');
    end
end
%% rescale: function description
function image = rescale(image)
    image_cell = num2cell(image,[1 2]);
    max_values = cellfun(@(x) max(x(:)), image_cell);
    image = uint8(bsxfun(@times, image, 255/max_values));
end
end