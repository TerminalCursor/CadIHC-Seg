#!/usr/bin/env python
# coding: utf-8
from utils import *
from cellpose import core, io, models, train

use_GPU = core.use_gpu()
assert use_GPU, "CUDA Unavailable. CPU Unadvisable"

# Setup Config
config = UnifiedConfig()
config.training.setup_cycles(5)

# Initialize Loggers
_logger = io.logger_setup()
logger = Logger("dump/")
logger.enabled = False

logger.addnl(config.tostr())

# Data Preparation
train_data, test_data = prepare_data(config.dataset.base_directory, tile=256)

consistent_kwargs = lambda cfg: {
    "rescale": False,
    "test_data": test_data[0],
    "test_labels": test_data[1],
    "save_path": "./" + "".join(logger.filename.split("/")[:-1]) + "/",
    "model_name": f"{logger.filename.split('/')[-1]}-{config.model.base_arch}",
    **cfg.training.cycle_kwargs(),
}

# Model Preparation
model = models.CellposeModel(**config.model.build_config.todict())

# Log Pre-training Performance
_time = 0
logtile(model, logger, train_data, test_data, time=_time)
logeval(model, logger, train_data, test_data, time=_time)
_time += 1

# Initially Overfit on the First Image
train.train_seg(
    model.net,
    train_data=[train_data[0][0]],
    train_labels=[train_data[1][0]],
    **consistent_kwargs(config),
)
logtile(model, logger, train_data, test_data, time=_time)
logeval(model, logger, train_data, test_data, time=_time)
_time += 1

# Train on the rest
train.train_seg(
    model.net,
    train_data=train_data[0],
    train_labels=train_data[1],
    **consistent_kwargs(config),
)
logtile(model, logger, train_data, test_data, time=_time)
logeval(model, logger, train_data, test_data, time=_time)
_time += 1

for _ in range(config.training.training_cycles):
    augmented_images, augmented_labels = generate_augmentations(train_data, config)
    for _ in range(config.training.n_gen):
        im, ma = generate_synthetic_stain()
        augmented_images.append(im.transpose(2, 0, 1))
        augmented_labels.append(ma)
    train.train_seg(
        model.net,
        train_data=augmented_images,
        train_labels=augmented_labels,
        **consistent_kwargs(config),
    )
    logtile(model, logger, train_data, test_data, time=_time)
    logeval(model, logger, train_data, test_data, time=_time)
    _time += 1
    config.training.next_cycle()

tiler_data = {
    "train": {
        idx: {
            "points": tiler.points,
            "point_data": tiler.point_meta,
        }
        for idx, tiler in enumerate(train_data[2])
    },
    "test": {
        idx: {
            "points": tiler.points,
            "point_data": tiler.point_meta,
        }
        for idx, tiler in enumerate(test_data[2])
    },
}

from numpy import save

save(f"{logger.filename}-tilers.npy", tiler_data)
