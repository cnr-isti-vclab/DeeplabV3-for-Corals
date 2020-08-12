from __future__ import print_function, division
import sys
import os
import numpy as np
from PIL import Image as PILimage
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from albumentations import (CLAHE, HueSaturationValue, RGBShift, RandomBrightnessContrast, Compose)




# ALBUMENTATIONS - USED JUST TO PERFORM THE COLOR AUGMENTATION
def augmentation_color(p=0.5):
    return Compose([

        CLAHE(clip_limit=3.0, tile_grid_size=(2, 2), always_apply=False, p=0.2),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=0.15, p=0.3),
        RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(0, 10), b_shift_limit=(0, 20), p=0.3),
        HueSaturationValue(hue_shift_limit=(0, 20), sat_shift_limit=0, val_shift_limit=0, p=0.3),
        ], p=p)


class CoralsDataset(Dataset):

    """Corals dataset."""

    def __init__(self, input_images_dir, input_labels_dir, dictionary, target_class):
        """
        :param input_images_dir: folder containing the images
        :param input_labels_dir: folder containing the labels
        :param dictionary: class-color dictionary
        :param target_classes: a dictionary containing the class under investigation
        """

        # IMAGES AND LABELS HAVE SAME NAMES BUT DIFFERENT DIRECTORIES
        self.images_dir = input_images_dir
        self.labels_dir = input_labels_dir
        self.images_names = [os.path.basename(x) for x in glob.glob(os.path.join(input_images_dir, '*.png'))]
        self.dict_colors = dictionary
        self.dict_target = target_class
        self.num_classes = len(target_class)

        # if background does not exists it is added
        self.dict_colors["Background"] = [0, 0, 0]

        # DATA LOADING SETTINGS
        self.flagDataAugmentation = True

        # DATA AUGMENTATION SETTINGS - GEOMETRY TRANSFORMS
        self.flagDataAugmentationFlip = True
        self.flagDataAugmentationRT = True
        self.RANDOM_TRANSLATION_MINVALUE = -50
        self.RANDOM_TRANSLATION_MAXVALUE = 50
        self.RANDOM_ROTATION_MINVALUE = -10
        self.RANDOM_ROTATION_MAXVALUE = 10

        self.flagDataAugmentationCrop = True
        self.CROP_SIZE = 513

        self.flagDataAugmentationScale = False
        self.RANDOM_SCALE_MIN = 0.9
        self.RANDOM_SCALE_MAX = 1.1

        # COLOR TRANSFORM
        self.custom_color_aug = augmentation_color(p=0.4)
        self.flagColorAugmentation = True

        # NORMALIZATION (True => average is removed)
        self.normalizationByRemoveAverage = True

        self.weights = None
        self.dataset_average = np.zeros(3, dtype=float)


    def augmentationSettings(self, range_T, range_R, range_scale, crop_size, augmentation_flip=True):
        """
        Set the augmentation parameters (of the geometric transformation).

        :param range_T: set the random translation in the range [-range_T, range_T] (in pixels)
        :param range_R: set the random rotation in the range [-range_R, range_R] (in degrees)
        :param range_scale: set the random scale in the range [1.0 - range_scale, 1.0 + range_scale]
        :param crop_size: set the center crop size to [crop_size, crop_size]
        :param augmentation_flip: enable/disable horizontal and vertical flip
        """

        self.flagDataAugmentationFlip = augmentation_flip

        if (range_T > 0.0) or (range_R > 0.0):
            self.flagDataAugmentationRT = True
            self.RANDOM_TRANSLATION_MINVALUE = -range_T
            self.RANDOM_TRANSLATION_MAXVALUE = range_T
            self.RANDOM_ROTATION_MINVALUE = -range_R
            self.RANDOM_ROTATION_MAXVALUE = range_R
        else:
            self.flagDataAugmentationRT = False

        if crop_size > 0:
            self.flagDataAugmentationCrop = True
            self.CROP_SIZE = crop_size
        else:
            self.flagDataAugmentationCrop = False

        if range_scale > 0.00001:
            self.flagDataAugmentationScale = True
            self.RANDOM_SCALE_MIN = 1.0 - range_scale
            self.RANDOM_SCALE_MAX = 1.0 + range_scale
        else:
            self.flagDataAugmentationScale = False

    def enableNormalizationByRemoveAverage(self):

        self.normalizationByRemoveAverage = True

    def enableAugumentation(self):

        self.flagDataAugmentation = True

    def disableAugumentation(self):

        self.flagDataAugmentation = False

    def enableColorAugmentation(self):

        self.flagColorAugmentation = True

    def disableColorAugmentation(self):

        self.flagColorAugmentation = False

# DA CAMBIARE

    def normalizeInputImage(self, image_tensor):
        """
        It normalizes the input image.
        :param image_tensor: image to normalize
        :return: normalized image
        """

        image_tensor[0] = image_tensor[0] - self.dataset_average[0]
        image_tensor[1] = image_tensor[1] - self.dataset_average[1]
        image_tensor[2] = image_tensor[2] - self.dataset_average[2]


        return image_tensor

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):

        # sample name
        sample_name = self.images_names[idx]

        img_filename = os.path.join(self.images_dir, self.images_names[idx])
        label_filename = os.path.join(self.labels_dir, self.images_names[idx])
        img = PILimage.open(img_filename)
        imglbl = PILimage.open(label_filename)

        # APPLY DATA AUGMENTATION
        if self.flagDataAugmentation:
            # SET COLOR TRANSFORMATION
            if self.flagColorAugmentation is True:
                img_np = np.array(img)
                data = {"image": img_np}
                augmented = self.custom_color_aug(**data)
                img_np = augmented["image"]
                img = PILimage.fromarray(img_np)



            # APPLY GEOMETRIC TRANSFORMATION

            # random flip
            img_flipped = img
            imglbl_flipped = imglbl

            if self.flagDataAugmentationFlip:
                # horizontal random flip
                if np.random.uniform() > 0.5:
                    img_flipped = img_flipped.transpose(PILimage.FLIP_LEFT_RIGHT)
                    imglbl_flipped = imglbl_flipped.transpose(PILimage.FLIP_LEFT_RIGHT)

                # vertical random flip
                if np.random.uniform() > 0.5:
                    img_flipped = img_flipped.transpose(PILimage.FLIP_TOP_BOTTOM)
                    imglbl_flipped = imglbl_flipped.transpose(PILimage.FLIP_TOP_BOTTOM)


            # rotation and translation

            if self.flagDataAugmentationRT:
                rot = np.random.randint(self.RANDOM_ROTATION_MINVALUE, self.RANDOM_ROTATION_MAXVALUE)
                tx = np.random.randint(self.RANDOM_TRANSLATION_MINVALUE, self.RANDOM_TRANSLATION_MAXVALUE)
                ty = np.random.randint(self.RANDOM_TRANSLATION_MINVALUE, self.RANDOM_TRANSLATION_MAXVALUE)
                img_flipped_RT = transforms.functional.affine(img_flipped, angle=rot, scale=1.0, shear=0.0,
                                                              translate=(tx, ty), resample=PILimage.BILINEAR)
                imglbl_flipped_RT = transforms.functional.affine(imglbl_flipped, angle=rot, scale=1.0, shear=0.0,
                                                                 translate=(tx, ty), resample=PILimage.NEAREST)
            else:
                img_flipped_RT = img_flipped
                imglbl_flipped_RT = imglbl_flipped


            # center crop
            if self.flagDataAugmentationCrop:
                w, h = img_flipped_RT.size
                left = (w / 2) - (self.CROP_SIZE / 2)
                top = (h / 2) - (self.CROP_SIZE / 2)
                img_augmented = transforms.functional.crop(img_flipped_RT, top, left, self.CROP_SIZE, self.CROP_SIZE)
                imglbl_augmented = transforms.functional.crop(imglbl_flipped_RT, top, left, self.CROP_SIZE,
                                                              self.CROP_SIZE)
            else:
                img_augmented = img_flipped_RT
                imglbl_augmented = imglbl_flipped_RT


            # PIL image -> Pytorch tensor
            img_tensor = transforms.functional.to_tensor(img_augmented)
            # normalize directly the Pytorch tensor
            img_tensor = self.normalizeInputImage(img_tensor)

            # PIL image -> Pytorch tensor
            imglbl_tensor = transforms.functional.to_tensor(imglbl_augmented)

            # create labels: from PIL image to Pytorch tensor
            labels_tensor = self.imageLabelToLongTensor(imglbl_augmented)

        else:

            # PIL image -> Pytorch tensor
            img_tensor = transforms.functional.to_tensor(img)
            # normalize directly the Pytorch tensor
            img_tensor = self.normalizeInputImage(img_tensor)

            # PIL image -> Pytorch tensor
            imglbl_tensor = transforms.functional.to_tensor(imglbl)

            # create labels: from PIL image to Pytorch tensor
            labels_tensor = self.imageLabelToLongTensor(imglbl)

        # image labels saves the label as image for check purposes
        sample = {'image': img_tensor, 'image_label': imglbl_tensor, 'labels': labels_tensor, 'name': sample_name}

        return sample


    def computeWeights(self):

        class_sample_count = np.zeros(self.num_classes)
        N = len(self.images_names)
        print(" ")
        for i, image_name in enumerate(self.images_names):

            label_filename = os.path.join(self.labels_dir, image_name)
            imglbl = PILimage.open(label_filename)
            data = np.array(imglbl)
            w = data.shape[1]
            h = data.shape[0]
            ox = int((w - self.CROP_SIZE) / 2)
            oy = int((h - self.CROP_SIZE) / 2)
            data_crop = data[oy:oy + self.CROP_SIZE, ox:ox + self.CROP_SIZE]

            labels = self.colorsToLabels(data_crop)
            existing_labels, counts = np.unique(labels, return_counts=True)

            for j in range(len(existing_labels)):
                class_sample_count[existing_labels[j]] += counts[j]

            sys.stdout.write("\rComputing frequencies... %.2f"% ((i * 100.0) / float(N)))

        # update target classes with the ones found in the dataset
        true_dict_target = dict()
        tot = np.sum(class_sample_count)
        temp_weights = []

        for key in self.dict_target.keys():
            index = self.dict_target[key]
            if class_sample_count[index] > 0:
                true_dict_target[key] = index
                temp_weights.append(tot / class_sample_count[index])

        # set indices progressively
        n = len(true_dict_target.keys())
        for index in range(n):
            if not index in true_dict_target.values():
                min_value = 10000
                for value in true_dict_target.values():
                    if value > index and value < min_value:
                        min_value = value

                for key in true_dict_target.keys():
                    if true_dict_target[key] == min_value:
                        true_dict_target[key] = index

        self.num_classes = len(temp_weights)
        self.weights = np.array(temp_weights)
        self.dict_target = true_dict_target


    def computeAverage(self):

        sum = np.zeros((self.CROP_SIZE, self.CROP_SIZE, 3), dtype=np.float)
        N = len(self.images_names)
        print(" ")
        for i, image_name in enumerate(self.images_names):

            img_filename = os.path.join(self.images_dir, image_name)
            img = PILimage.open(img_filename)
            data = np.array(img, dtype=np.float)
            w = data.shape[1]
            h = data.shape[0]
            ox = int((w - self.CROP_SIZE) / 2)
            oy = int((h - self.CROP_SIZE) / 2)
            data_crop = data[oy:oy + self.CROP_SIZE, ox:ox + self.CROP_SIZE]
            sum += data_crop

            sys.stdout.write("\rComputing average... %.2f"% ((i * 100.0) / float(N)))

        img_mean = sum / len(self.images_names)
        self.dataset_average[0] = np.mean(img_mean[:,:,0]) / 255.0
        self.dataset_average[1] = np.mean(img_mean[:,:,1]) / 255.0
        self.dataset_average[2] = np.mean(img_mean[:,:,2]) / 255.0

    def colorsToLabels(self, data):
        """
        It converts the colors stored in a numpy array to the labels.
        """
        # array NumPy.
        height = data.shape[0]
        width = data.shape[1]
        labelsint = np.zeros((height, width), dtype='int64')
        labelsint[:] = self.dict_target['Background']

        for key in self.dict_target.keys():
            colors = self.dict_colors[key]
            idx = np.where((data[:, :, 0] == colors[0]) & (data[:, :, 1] == colors[1]) & (data[:, :, 2] == colors[2]))
            labelsint[idx] = self.dict_target[key]

        return labelsint


    def imageLabelToLongTensor(self, image_label):
        """
        It converts an image label to a Pytorch Long Tensor containing the class labels.

        :param image_label: input image is a PIL image
        :param image_label_mask:  label mask. It is applied only if the masking flag is True.
        :return: Pytorch Long Tensor
        """

        data = np.array(image_label)
        height = data.shape[0]
        width = data.shape[1]
        labelsint = np.zeros((height, width), dtype='int64')
        labelsint[:] = self.dict_target['Background']

        for key in self.dict_target.keys():
            colors = self.dict_colors[key]
            idx = np.where((data[:, :, 0] == colors[0]) & (data[:, :, 1] == colors[1]) & (data[:, :, 2] == colors[2]))
            labelsint[idx] = self.dict_target[key]

        labels_t = torch.from_numpy(labelsint)

        return labels_t


    def show(self, i):
        """
        It shows the i-th sample of the dataset.
        :param idx: index of the dataset element
        """

        sample = self[i]
        print(sample['name'])
        plt.figure(1)
        plt.imshow(sample['image'].numpy().transpose(1, 2, 0))
        plt.figure(2)
        plt.imshow(sample['image_label'].numpy().transpose(1, 2, 0))
        plt.figure(3)
        plt.imshow(sample['labels'].numpy())
        plt.show()


    def saveClassificationResult(self, img_tensor, output_tensor, filename):
        """
        It saves two images showing the classification result and the overlay with the input image.

        :param img_tensor: input image (as a Pytorch Tensor with 3 channels)
        :param output_tensor: Pytorch Float Tensor [N-1 x 224 x 224] (N classes)
        :param filename: full name of the image to save
        """

        values, pred_indices_t = torch.max(output_tensor, 0)

        pred_indices = pred_indices_t.numpy()

        img = np.zeros((pred_indices.shape[0], pred_indices.shape[1], 3), dtype='uint8')

        class_names = list(self.dict_target)

        for i in range(pred_indices.shape[0]):
            for j in range(pred_indices.shape[1]):
                label = pred_indices[i][j]

                color_name = class_names[label]
                color = self.dict_colors[color_name]

                img[i][j][0] = color[0]
                img[i][j][1] = color[1]
                img[i][j][2] = color[2]

        # classification map
        image_class = PILimage.fromarray(img, 'RGB')
        image_class.save(filename, format="PNG")

