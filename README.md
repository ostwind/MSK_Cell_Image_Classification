# Hodgkin's Lymphoma Cell Classification

Tumor tissues and images were procured by the Ingo Mellinghoff Lab at Memorial Sloan Kettering Cancer Center as part of on-going oncology research. As such, the images are not publicly available. 

Our machine learning task may be formulated as: given a multi-channel cell image where individual channels correspond to experimentally introduced biological markers, classify the cell between tumor/T-cell types, and if possible, the specific T-cell type.

See report.pdf for details 

## Data

![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/pipeline.png)

Right angle rotations are applied to individual cell samples as data augmentation. Zero mean and unit variance normalization are also applied to each image. Images below correspond to the same sample viewed through different immunofluorescence markers. The values above denote class probabilities. 

![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/4-13156_other_1110_CD3_CD20_S029.png)
![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/4-13156_other_1110_CD3_CD4_S029.png)
![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/4-13156_other_1110_CD3_CD8_S029.png)

## Model

The model is an implementation of the Ladder Network as described in [Rasmus et. al.](https://arxiv.org/abs/1507.02672). The pipeline from sample generation to postprocessing can be found under MSK_Cell_Image_Classification/src/ 
![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/network_archi.png)

View on tensorboard:
![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/ladder_network.png)

## Research UI

To reduce the cost of generating our first labeled dataset, oncology researchers are prompted to proceed through the labeling process via an user-interface. This is found at MSK_Cell_Image_Classification/src/UI/run_UI.py. To assist in inputting images into HALO proprietary software, an automatic .afi writer is found in MSK_Cell_Image_Classification/src/HALO_util/afi.py.

![alt text](https://github.com/ostwind/MSK_Cell_Image_Classification/blob/master/figures/UI_snapshot.png)

