#!/usr/bin/env python
# coding: utf-8
from utils import *
from numpy.ma import masked_equal
from matplotlib import pyplot as plt
from cellpose import models
from tqdm.auto import tqdm

config = UnifiedConfig()
config.pretrained("promising/Model-C00.pt")

train_data, test_data = prepare_data(config.dataset.base_directory)

model = models.CellposeModel(**config.model.build_config.todict())

ex_im = test_data[0][0]
ex_fx = postforward(model, ex_im)

mask = color_mask(ex_fx)

plt.imsave("tmp_out/ex/fixed.png", (masked_equal(ex_fx, 0)))
plt.imsave("tmp_out/ex/fixed-cmap.png", mask)
if len(mask.shape) == 2:
    plt.imsave("tmp_out/ex/fixed-cmap-jet.png", mask, cmap="jet")
plt.imsave("tmp_out/ex/fixed-im.png", ex_im.transpose(1, 2, 0))
