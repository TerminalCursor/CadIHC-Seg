#!/usr/bin/env python
# coding: utf-8
from histseg.utils import *
from cellpose import core, io, models, train
from copy import deepcopy as dc
from numpy import array

use_GPU = core.use_gpu()
assert use_GPU, "CUDA Unavailable. CPU Unadvisable"

# Setup Config
model_config = UnifiedConfig()
model_config.training.start_learning_rate = 5e-5
model_config.training.end_learning_rate = 5e-6
model_config.training.epochs = 25
model_config.training.setup_cycles(10)

# Initialize Loggers
_logger = io.logger_setup()
log = Logger("dump/")

log.addnl(model_config.tostr())

# Data Preparation
c_train_data, c_test_data = prepare_data(model_config.dataset.base_directory, tile=None)

consistent_kwargs = lambda cfg: {
    "rescale": False,
    "test_data": test_data[0],
    "test_labels": test_data[1],
    "save_path": "./" + "".join(log.filename.split("/")[:-1]) + "/",
    "model_name": f"{log.filename.split('/')[-1]}-{model_config.model.base_arch}",
    **cfg.training.cycle_kwargs(),
}

# Model Preparation
model = models.CellposeModel(**model_config.model.build_config.todict())
log.addnl(f"{model.net}")

# Log Pre-training Performance
_time = 0
train_data = dc(c_train_data)
test_data = dc(c_test_data)
logeval(model, log, train_data, test_data, time=_time)
_time += 1

# Initially Overfit on the First Image
del train_data
del test_data
train_data = dc(c_train_data)
test_data = dc(c_test_data)
train.train_seg(
    model.net,
    train_data=[train_data[0][0]],
    train_labels=[train_data[1][0]],
    **consistent_kwargs(model_config),
)
train_data = dc(c_train_data)
test_data = dc(c_test_data)
logeval(model, log, train_data, test_data, time=_time)
_time += 1

# Train on the rest
del train_data
del test_data
train_data = dc(c_train_data)
test_data = dc(c_test_data)
train.train_seg(
    model.net,
    train_data=train_data[0],
    train_labels=train_data[1],
    **consistent_kwargs(model_config),
)
train_data = dc(c_train_data)
test_data = dc(c_test_data)
logeval(model, log, train_data, test_data, time=_time)
_time += 1

for _ in range(model_config.training.training_cycles):
    del train_data
    del test_data
    train_data = dc(c_train_data)
    test_data = dc(c_test_data)
    augmented_images, augmented_labels = generate_augmentations(train_data, model_config)
    for _ in range(model_config.training.n_gen):
        im, ma = generate_synthetic_stain()
        augmented_images.append(im.transpose(2, 0, 1))
        augmented_labels.append(ma)
    train.train_seg(
        model.net,
        train_data=augmented_images,
        train_labels=augmented_labels,
        **consistent_kwargs(model_config),
    )
    del augmented_images
    del augmented_labels
    del train_data
    del test_data
    train_data = dc(c_train_data)
    test_data = dc(c_test_data)
    logeval(model, log, train_data, test_data, time=_time)
    _time += 1
    model_config.training.next_cycle()
