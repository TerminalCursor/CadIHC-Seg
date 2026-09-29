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

cn_data, cn_maps = full_energetics(
    "promising/Model-C00.pt", "promising/Model-N00.pt", ex_im, fixup=True
)

for idx in range(len(cn_maps)):
    name = ["Cell", "Nuclear"][idx]
    maps = cn_maps[idx]
    maps = masked_equal(maps, 0)
    plt.imsave(f"tmp_out/en/{name}-Pressure.png", maps[..., 0])
    plt.imsave(f"tmp_out/en/{name}-Tension.png", maps[..., 1])
    plt.imsave(f"tmp_out/en/{name}-CC-Adhesion.png", maps[..., 2])
    plt.imsave(f"tmp_out/en/{name}-CT-Adhesion.png", maps[..., 3])

for idx in range(cn_maps[0].shape[-1]):
    name = ["Pressure", "Tension", "CC-Adhesion", "CT-Adhesion"][idx]
    c_maps = cn_maps[0][..., idx]
    n_maps = cn_maps[1][..., idx]
    maps = c_maps + n_maps
    plt.imsave(f"tmp_out/en/Together-{name}.png", maps)
