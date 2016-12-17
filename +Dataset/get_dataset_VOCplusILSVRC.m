function get_dataset_VOCplusILSVRC(usage, use_flip, varargin)
% Pascal voc 2012 trainval set
% set opts.imdb_train opts.roidb_train
% or set opts.imdb_test opts.roidb_train

% -----set dataset to global variable to save memory -----%
global dataset
ip = inputParser;
ip.addParameter('dataroot',       '/disks/sda/01/Yifan_sda/image_data', @ischar);
ip.addParameter('imdb_cache_dir', '.',                                @ischar);
ip.parse(varargin{:});
opts = ip.Results;
% change to point to your devkit install
ilsvrc_root     = fullfile(opts.dataroot, 'ILSVRC2014_devkit');
voc_2012_root   = fullfile(opts.dataroot,'VOCdevkit2012');
voc_2007_root   = fullfile(opts.dataroot, 'VOCdevkit2007');

switch usage
    case {'train'}
        dataset.imdb_train    = { ...
            imdb_from_voc(voc_2012_root, 'trainval', '2012', opts.imdb_cache_dir, use_flip); ...
            imdb_from_voc(voc_2007_root, 'trainval', '2007', opts.imdb_cache_dir, use_flip); ...
            imdb_from_ilsvrc(ilsvrc_root, 'train', use_flip, opts.imdb_cache_dir, 800) ...
            };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, 'rootDir', opts.imdb_cache_dir), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = { ...
            imdb_from_voc(voc_2007_root, 'test', '2007', opts.imdb_cache_dir, use_flip); ...
            imdb_from_ilsvrc(ilsvrc_root, 'test', false, opts.imdb_cache_dir); ...
        };
        dataset.roidb_test    = cellfun(@(x) x.roidb_func(x, 'rootDir', opts.imdb_cache_dir), dataset.imdb_test, 'UniformOutput', false);
    case {'val'}
        dataset.imdb_test     = { ...
            imdb_from_voc(voc_2007_root, 'test', '2007', opts.imdb_cache_dir, use_flip); ...
            imdb_from_ilsvrc(ilsvrc_root, 'val', false, opts.imdb_cache_dir); ...
        };
        dataset.roidb_test    = cellfun(@(x) x.roidb_func(x, 'rootDir', opts.imdb_cache_dir), dataset.imdb_test, 'UniformOutput', false);
    otherwise
        error('usage = ''train'' or ''test'' or ''val'' ');
end
remove_empty_gt()

    %% remove_non_gt: remove roidb and imdb with empty gt, since this causes problem in sampling rois during training
    function remove_empty_gt()
        switch usage
        case {'train'}
            for j = 1:length(dataset.roidb_train)
                hasGT = arrayfun(@(x) any(x.gt), dataset.roidb_train{j}.rois, 'UniformOutput', true);
                if all(hasGT)
                    continue;
                end
                dataset.roidb_train{j}.rois(~hasGT)         = [];
                dataset.imdb_train{j}.image_ids(~hasGT)     = [];
                dataset.imdb_train{j}.sizes(~hasGT,:)         = [];
                imdb = dataset.imdb_train{j};
                imdb.image_at = @(i) sprintf('%s/%s.%s',imdb.image_dir,imdb.image_ids{i},imdb.extension);
                dataset.imdb_train{j} = imdb;
                save(imdb.cache_file, 'imdb');
                roidb = dataset.roidb_train{j};
                save(roidb.cache_file, 'roidb');
            end
        otherwise
            for j = 1:length(dataset.roidb_test)
                hasGT = arrayfun(@(x) any(x.gt), dataset.roidb_test{j}.rois, 'UniformOutput', true);
                if all(hasGT)
                    continue;
                end
                dataset.roidb_test{j}.rois(~hasGT)         = [];
                dataset.imdb_test{j}.image_ids(~hasGT)     = [];
                dataset.imdb_test{j}.sizes(~hasGT,:)         = [];
                imdb = dataset.imdb_test{j};
                imdb.image_at = @(i) sprintf('%s/%s.%s',imdb.image_dir,imdb.image_ids{i},imdb.extension);
                dataset.imdb_test{j} = imdb;
                save(imdb.cache_file, 'imdb');
                roidb = dataset.roidb_test{j};
                save(roidb.cache_file, 'roidb');
            end
        end
    end
end
