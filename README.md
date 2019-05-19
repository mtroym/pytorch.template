# Pytorch training framework.- (K)Teras
pytorch1.0.0 training framework - in progress
## ~~TorchLayers Library (Tony version.)~~ -- depracated
- [x] ~~Separable Convolution~~
- [x] ~~ConvLSTM~~
- [x] ~~Bi-Directional ConvLSTM~~

## models.
- [x] ~~Xception (Atrous Version)~~ (Use other version. to make sure if it works.)
- [x] ~~ASPP (Atrous Spatial Pyramids Pooling)~~
- [x] srcnn - Super Resolution task.
- [x] ~~deeplab v3+ - Object Segmentation task~~
- [x] deepLab3D custom model.
- [x] deepLabZ custom model. 
- [ ] PSPNet for semantic segmentation.
- [ ] CRF/MRF - post  processing for object segmentation task.
- [ ] Bi-AtrousLSTM (Bi-Directional Atrous LSTM//cascaded LSTM**)


## datasets.
- [x] CelebA - face dataset
- [x] VOCParts - detection dataset
- [x] segTHOR - medical image segmentation dataset.
- [ ] IMaterialistFashion For Some challenge.
- [ ] VOC2012/2007 - object detection/segmentation dataset
- [ ] DAVIS 2017 - video segmentation dataset

## eval metrics/criterion
- [x] `mIoU` 
- [x] `IoU` , contains `mIoU`
- [x] segTHOR's. Hausdorff/ Dice from `SimpleITK` package(3D).


## TODOS:
- [ ] Change criterion and metrics to 2 single folders.
- [ ] make visualization separately.
- [ ] make file structure more clear.
- [ ] Make  a  `keras` version.


## Time modified.
- Last Modified: May 19 11:33PM Tony. 