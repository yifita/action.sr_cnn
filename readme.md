### BMVC2016 [Two-Stream SR-CNNs for Action Recognition in Videos](http://www.bmva.org/bmvc/2016/papers/paper108/index.html)

---
### Prerequisites ###

#### Caffe
- clone and build caffe from [here](https://github.com/yifita/caffe). This caffe version contains `merge_batch` and `weighted_sum` layer. In addition it exposed some protected caffe functions in the matlab interface to emulate `iter_size` in matlab.
- modify caffe_mex.m to the corresponding caffe matlab interface directory

#### Optical Flow
- extract optical flow with Limin's [flow extractor](https://github.com/wanglimin/dense_flow)

#### Bounding Boxes
- We extracted 118 objects' bounding boxes in all video frames using [Faster-RCNN][Faster-RCNN] (retraining is required) and obtained filtered bounding boxes taking consideration of temporal coherency and motion saliency.
- The extracted and processed bounding boxes for [ucf-101][ucf-101] can be downloaded [here](https://polybox.ethz.ch/index.php/s/fNPgASRZiaVYsrr). Place the downloaded mat files under `imdb/cache`.
- If you wish to extract the bounding boxes yourself, you need to be able to run Ren Shaoqing's [Faster-RCNN][Faster-RCNN] (most codes are migrated into this repository with minor modifications and more comments)
	- First generate raw object detection using `faster_rcnn_{dataset}.m` 
	- Then use `action/prepare_rois_context.m` to process bounding boxes as described in the paper.

---
### Test ###
#### datasets ####
create dataset.mat using `imdb/get_{name}_dataset.m` (Directories may need to be adjusted!)
An example of generated ucf_dataset.mat 
#### models ####
- `models/srcnn/{stream}` contains model prototxt files
- model weights can be downloaded in the following links

	| Stream        | person+scene (the final proposed model in the paper)  |
	| ------------- |:-------------:|
	| spatial      | [split1](https://polybox.ethz.ch/index.php/s/sw6XuddNvN0UsDb) [split2](https://polybox.ethz.ch/index.php/s/xOkENBiQ6ItPjkc) [split3](https://polybox.ethz.ch/index.php/s/HCSFWRmYdgeEECH) |
	| flow      	 | [split1](https://polybox.ethz.ch/index.php/s/IXxAciMJ2eJE2U7) [split2](https://polybox.ethz.ch/index.php/s/5gNrgpKrwR35mMm) [split3](https://polybox.ethz.ch/index.php/s/Jk58PgHVbVrNfFl) |

- the reported two-stream results in the paper are yielded from summing spatial and temporal classification scores using weight 1 : 3.
- other models mentioned in the paper experiments can be provided if the demand is large.

#### run ####
in matlab 
```matlab
% test spatial
test_spatial('model_path', path_to_weights, 'split', 1)
```

```matlab
% test flow
`test_flow('model_path', path_to_weights, 'split', 1)`
```

[ucf-101]: http://crcv.ucf.edu/data/UCF101.php
[hmdb-51]: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[jhmdb]: http://jhmdb.is.tue.mpg.de/
[Faster-RCNN]: faster_rcnn_build
