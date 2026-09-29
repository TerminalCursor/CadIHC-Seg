#!/usr/bin/env python
# coding: utf-8
from utils import *
from numpy.ma import masked_equal
from matplotlib import pyplot as plt
from cellpose import models
from tqdm.auto import tqdm

config = UnifiedConfig()

train_data, test_data = prepare_data(config.dataset.base_directory)

ex_im = test_data[0][0]

cm_cfg = ModelBuildConfig()
cm_cfg.pretrained_model = "promising/Model-C00.pt"
cm = models.CellposeModel(**cm_cfg.todict())

from utils.postprocess import postforward

# Cellpose without filling in holes
cm_out = cm.eval([ex_im])
cm_ma = cm_out[0][0]
cm_raw = cm_out[1][0]
cm_cprob = cm_out[1][0]

# Cellpose after filling in holes
cm_fixup = postforward(cm, ex_im)

plt.imsave(f"tmp_out/CS-S0.png", color_mask(cm_ma))
plt.imsave(f"tmp_out/CS-SX.png", color_mask(cm_ma), cmap="cool")
plt.imsave(f"tmp_out/CS-RW.png", cm_raw[0])
plt.imsave(f"tmp_out/CS-PR.png", cm_cprob[2], cmap="inferno")
plt.imsave(f"tmp_out/CS-S1.png", color_mask(cm_fixup))
plt.imsave(f"tmp_out/CS-SY.png", color_mask(cm_fixup), cmap="cool")
