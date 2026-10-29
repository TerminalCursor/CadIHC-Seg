#!/usr/bin/env python
# coding: utf-8
from histseg.utils import *
from json import dumps
dump = lambda x: dumps(x, indent=2)

DATASET_BASE_DIRECTORY = "ihc/data/segmentation/histology-250929/"

use_GPU = core.use_gpu()
assert use_GPU, "CUDA Unavailable. CPU Unadvisable"

# 1. Load Model
model = load_pth("promising/Model-C00.pt")
print("Loaded model")
print(model.net)

# 2. Load Data
# 2.a. Subsample to 10x
# 2.b. Adjust contrasts
#train_data, test_data = prepare_data("ihc/data/segmentation/histology-250929/", tile=None)
train_data, test_data = prepare_data(DATASET_BASE_DIRECTORY, tile=None)
#out_tr = list(zip(*train_data))
#out_te = list(zip(*test_data))
train_images = train_data[0]
train_labels = train_data[1]
test_images = test_data[0]
test_labels = test_data[1]
print("Loaded data")
print(f"Train: {len(train_images)} {len(train_labels)}")
print(f"Test: {len(test_images)} {len(test_labels)}")

# 3. Segment
train_segmen = postforward(model, train_images)
test_segmen = postforward(model, test_images)
print("Segmented")
print(f"Train: {len(train_segmen)}")
print(f"Test: {len(test_segmen)}")

# 4. Evaluate: Dice and IoU
train_evaluation = evaluate_model_m(train_labels, train_segmen)
test_evaluation = evaluate_model_m(test_labels, test_segmen)
print("Evaluation")
print(f"Train:")
print(dump(train_evaluation))
print(f"Test:")
print(dump(test_evaluation))
