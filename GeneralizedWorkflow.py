#!/usr/bin/env python
# coding: utf-8
from utils import *
from cellpose import core, io, models, train
from copy import deepcopy as dc
from numpy import array

use_GPU = core.use_gpu()
assert use_GPU, "CUDA Unavailable. CPU Unadvisable"

# Setup Config
config = UnifiedConfig()
config.training.start_learning_rate = 5e-5
config.training.end_learning_rate = 5e-6
config.training.epochs = 25
config.training.setup_cycles(10)

# Initialize Loggers
_logger = io.logger_setup()
logger = Logger("dump/")
#logger.enabled = False

logger.addnl(config.tostr())

# Data Preparation
#train_data, test_data = prepare_data(config.dataset.base_directory, tile=256)
c_train_data, c_test_data = prepare_data(config.dataset.base_directory, tile=None)

def square_tiles(data_tuple_list, d=512):
    from numpy import floor, ceil, linspace
    from itertools import product
    ims = data_tuple_list[0]
    mas = data_tuple_list[1]
    assert len(ims) == len(mas)
    nims = []
    nmas = []
    for idx in range(len(ims)):
        im = ims[idx]
        ma = mas[idx]
        nx = int(ceil(im.shape[1]/float(d)))
        ny = int(ceil(im.shape[2]/float(d)))
        xs = floor(linspace(0,im.shape[1]-d,nx)).astype(int)
        ys = floor(linspace(0,im.shape[2]-d,nx)).astype(int)
        assert im.shape[0] <= 4
        for x, y in product(xs, ys):
            sim = im[:,x:x+d,y:y+d]
            sma = ma[x:x+d,y:y+d]
            nims.append(sim)
            nmas.append(sma)
    return (nims, nmas)

c_train_data = square_tiles(c_train_data)

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
logger.addnl(f"{model.net}")

# Log Pre-training Performance
_time = 0
#logtile(model, logger, train_data, test_data, time=_time)
train_data = dc(c_train_data)
test_data = dc(c_test_data)
logeval(model, logger, train_data, test_data, time=_time)
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
    **consistent_kwargs(config),
)
#logtile(model, logger, train_data, test_data, time=_time)
train_data = dc(c_train_data)
test_data = dc(c_test_data)
logeval(model, logger, train_data, test_data, time=_time)
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
    **consistent_kwargs(config),
)
train_data = dc(c_train_data)
test_data = dc(c_test_data)
#logtile(model, logger, train_data, test_data, time=_time)
logeval(model, logger, train_data, test_data, time=_time)
_time += 1

for _ in range(config.training.training_cycles):
    del train_data
    del test_data
    train_data = dc(c_train_data)
    test_data = dc(c_test_data)
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
    #logtile(model, logger, train_data, test_data, time=_time)
    del augmented_images
    del augmented_labels
    del train_data
    del test_data
    train_data = dc(c_train_data)
    test_data = dc(c_test_data)
    logeval(model, logger, train_data, test_data, time=_time)
    _time += 1
    config.training.next_cycle()

#train_data = dc(c_train_data)
#test_data = dc(c_test_data)
#tiler_data = {
#    "train": {
#        idx: {
#            "points": tiler.points,
#            "point_data": tiler.point_meta,
#        }
#        for idx, tiler in enumerate(train_data[2])
#    },
#    "test": {
#        idx: {
#            "points": tiler.points,
#            "point_data": tiler.point_meta,
#        }
#        for idx, tiler in enumerate(test_data[2])
#    },
#}
#
#from numpy import save
#
#save(f"{logger.filename}-tilers.npy", tiler_data)
