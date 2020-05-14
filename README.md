# Deeplab-for-Corals

This code is part of a bigger project about the semantic segmentation of coral reefs for ecological monitoring. We use the DeepLab V3+ for the segmentation of ortho-mosaics and ortho-projections of seabed. The implementation of the Deeplab is based on the [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) project.

Strong points:

- Several loss functions are available for the training. 
- Default hyperparameters works reasonably well in many different cases.
- Minimal configuration for re-use.

## Requirements 

requirement goes here..

## Dataset

Our data are provided by the Scripps Institute of Oceanography. 

## Dataset preparation

Since our goal is to work with ortho-mosaics, the input orthoimage, should be subdivided into sub-images (tiles) before to be used 
for the training. The resolution of the input tiles should be greater than 513 x 513. The ground truth segmentation has to be provided as a color map with the same name of the corresponding image tile (24 bit, RGB) stored in a different folder.

For example:

 ```
dataset/image_tiles/tile0000.png
dataset/image_tiles/tile0001.png
dataset/image_tiles/tile0002.png
..
dataset/label_tiles/tile0000.png
dataset/label_tiles/tile0001.png
dataset/label_tiles/tile0002.png
..
```

To associate the class names with the labels color you need to modify `labelsdictionary.py`.
The dataloader automatically performs all the operations on-the-fly required by the training. 

## Training

The training parameters (number of epochs, learning rate, etc.) can be found at the beginning of the __main__ in `training.py`


## Team

Deeplab-for-Corals has been developed by Gaia Pavoni (gaia.pavoni@isti.cnr.it) and Massimiliano Corsini (massimiliano.corsini@isti.cnr.it). 
Feel free to contact us for any inquiries about the project.
