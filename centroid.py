#!/usr/bin/env python
# coding: utf-8
from utils import *
from matplotlib import pyplot as plt
from cellpose import models

config = UnifiedConfig()


def load_image(ffile):
    image = plt.imread(ffile)
    assert len(image.shape) == 3
    if image.shape[0] > 4:
        image = image.transpose(2, 0, 1)
    if image.shape[0] == 4:
        image = image[:3, ...]
    image = transform_image(image)
    return image


im = load_image("../BaseImage.png")
# Cellpose without filling in holes
cm_cfg = ModelBuildConfig()
cm_cfg.pretrained_model = "promising/Model-C00.pt"
cm = models.CellposeModel(**cm_cfg.todict())
cm_out = cm.eval([im])
cm_ma = cm_out[0][0]

out_im = im.copy()
plt.imsave("../ma.png", color_mask(cm_ma))
from scipy.ndimage import center_of_mass

names = list(
    set([int(v) for v in np.unique(cm_ma).tolist()])
    - {
        0,
    }
)
data = []
d = 4
for v in names:
    x, y = center_of_mass(cm_ma == v)
    X = int(x)
    Y = int(y)
    x = float(x) / cm_ma.shape[0]
    y = float(y) / cm_ma.shape[1]
    out_im[0, X : X + d, Y : Y + d] = 1
    out_im[1:, X : X + d, Y : Y + d] = 0
    data.append((x, y))
data = np.array(data)
plt.imsave("../BaseImage_Cell.png", out_im.transpose(1, 2, 0))
np.savetxt("../BaseImage.csv", data)
