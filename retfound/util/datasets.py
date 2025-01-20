# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Define a custom ImageFolder class to return the file names of the images
class ImageFolderWithPaths(datasets.ImageFolder):
    # Override the __getitem__ method to return image file paths as well
    def __getitem__(self, index):
        # Original method from ImageFolder
        sample, target = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]  # The image file path
        image_name = os.path.basename(path)  # Extract image name from path
        return sample, target, image_name  # Return image, label, and image name
    

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = ImageFolderWithPaths(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN # [0.485, 0.456, 0.406]
    std = IMAGENET_DEFAULT_STD # [0.229, 0.224, 0.225]
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic', # interesting! shall we use bicubic in saliency map?
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    # Crop: if input size is <= 224, use 256 as the minimum size for center crop; else, use the input size, no crop
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0 # full size
    size = int(args.input_size / crop_pct) # resize using bicubic
    t.append(
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size)) # ensure center 
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std)) # normalize using ImageNet values
    return transforms.Compose(t) # return a composed transform
