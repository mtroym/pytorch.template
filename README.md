# Ignite_pth
pytorch1.0.0 training framework - in progress
## TorchLayers Library (Tony version.)
- [x] Separable Convolution
- [x] ConvLSTM
- [x] Bi-Directional ConvLSTM

## models.
- [x] ~~Xception (Atrous Version)~~ (Use other version. to make sure if it works.)
- [x] ~~ASPP (Atrous Spatial Pyramids Pooling)~~
- [x] srcnn - Super Resolution task.
- [x] ~~deeplab v3+ - Object Segmentation task~~
- [ ] CRF/MRF - post  processing for object segmentation task.
- [ ] Bi-AtrousLSTM (Bi-Directional Atrous LSTM//cascaded LSTM**)
## datasets.
- [x] CelebA - face dataset
- [x] VOCParts - detection dataset
- [x] segTHOR - medical image segmentation dataset.
- [ ] VOC2012/2007 - object detection/segmentation dataset
- [ ] DAVIS 2017 - video segmentation dataset

## eval metrics/criterion
- [x] mIoU 
- [x] IoU , contains mIoU
- [x] segTHOR's. Hausdorff/ Dice from SimpleITK package. 