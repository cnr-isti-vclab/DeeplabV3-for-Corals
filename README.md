# Corals Semantic Segmentation

This code is part of a bigger project about the semantic segmentation of coral reefs for ecological monitoring. We use the DeepLab V3+ for the segmentation of ortho-mosaics and ortho-projections of seabed. The implementation of the Deeplab is based on the [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) project.

Strong points:

- Several loss functions are available for the training. 
- Default hyperparameters works reasonably well in many different cases.
- Minimal configuration for re-use.

In particular, we re-implemented the [Generalized Dice Loss (GDL)](https://arxiv.org/abs/1707.03237) and the [Boundary Loss](https://github.com/LIVIAETS/surface-loss).

## Requirements 

This code has been tested on Python 3.6 and with different versions of CUDA (both 9.2 and 10.0) and Pytorch (>= 1.2).

| Module name  | Version |
| ------------- | ------------- |
| albumentations | 0.2.2 |
| numpy | 1.16.4 |
| opencv-python | 4.1.0.25 |
| Pillow | 5.2.0 |
| scikit-image | 0.15.0 |
| scikit-learn | 0.21.2 |
| scipy | 1.4.1 |
| tensorboard | 2.1.1 |
| tensorflow | 2.1.0 |
| torch  | 1.5.0  |
| torchvision | 0.6.0 |


## Dataset

Our data are provided by the Scripps Institute of Oceanography. 

## Dataset preparation

Since our goal is to work with ortho-mosaics, typically in the order of hundreds of MPixels, a subdivision of the input orthoimage into sub-images (tiles) is necessary. Each tile is cropped to 513 x 513 and feed to the network. Hence the resolution of the input tiles should be greater than 513 x 513. The ground truth segmentation has to be provided as a color map with the same name of the corresponding image tile (24 bit, RGB) stored in a different folder.

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
The dataloader automatically performs on-the-fly all the operations required by the training (geometric augmentation, color augmentation, and cropping).

## Training

Our training is based on fine-tuning the version of the DeepLab just mentioned. So, you need to download the pre-trained weights from [here](https://drive.google.com/file/d/1i57CxuVgD1I0P5qI1gtUvKi45ymull_x/view?usp=sharing).

The training parameters (number of epochs, learning rate, etc.) and theirs description can be found at the beginning of the __main__ in `training.py` . After the settings are ready the training can be launched by simply:


 ```
python training.py
..
```


## Team

DeeplabV3-for-Corals has been developed by Gaia Pavoni (gaia.pavoni@isti.cnr.it) and Massimiliano Corsini (massimiliano.corsini@isti.cnr.it). 
Feel free to contact us for any inquiries about the project.
