#!/usr/bin/env python
# coding: utf-8
from utils import *
from cellpose import core, io, models, train

# Data Preparation
train_data, test_data = prepare_data("histology-250929/")

config = UnifiedConfig()
augmented_images, augmented_labels = generate_augmentations(train_data, config)
for _ in range(config.training.n_gen):
    im, ma = generate_synthetic_stain(_debug=True)
    augmented_images.append(im.transpose(2, 0, 1))
    augmented_labels.append(ma)

def fixup(x):
    if len(x.shape) == 2:
        return color_mask(x)
    if x.shape[0] <= 4:
        return x.transpose(1,2,0)
    return x

for idx in range(len(augmented_labels)):
    ncoloring = fixup(augmented_labels[idx])
    plt.imsave(f"train-im-samples/{idx}im.png", fixup(augmented_images[idx]))
    plt.imsave(f"train-im-samples/{idx}ma.png", ncoloring)
    #im = augmented_images[idx].copy()
