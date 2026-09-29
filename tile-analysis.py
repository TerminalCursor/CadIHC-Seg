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

tile_data_f = "./dump/20251204-095934-499860.txt-tilers.npy"
tile_data = np.load(tile_data_f, allow_pickle=True).item()

VMIN = 0.5
TSIZE = 10

data = {
    "train": train_data,
    "test": test_data,
}


def process(td, d, tt, im, proc=None, tile_w=256, tile_h=256):
    from numpy import array, zeros

    assert tt in ["train", "test"]
    image = d[tt][0][im]
    if image.shape[0] <= 4:
        image = image.transpose(1, 2, 0)
    mask = d[tt][1][im]
    tile_data = td[tt][im]
    out_data = zeros((mask.shape[0], mask.shape[1]))
    points = tile_data["points"]
    point_data = tile_data["point_data"]
    point_idx_list = range(points.shape[0])
    # print(array([array(point_data[point_idx]).squeeze()[-1,0] for point_idx in point_idx_list]).min())
    assert proc is not None
    for point_idx in point_idx_list:
        point = points[point_idx]
        x, y = point
        data = array(point_data[point_idx]).squeeze()
        v = proc(data)
        out_data[x : x + tile_w, y : y + tile_h] = v
    return out_data


def proc(x):
    # Get AJI
    # x = x[:,0]
    # Get Dice
    x = x[:, 1]
    return x.max()


procs = [
    # ("JI-Max", lambda x: x[:,0].max()),
    # ("F1-Max", lambda x: x[:,1].max()),
    # ("JI-Mean", lambda x: x[:,0].mean()),
    # ("F1-Mean", lambda x: x[:,1].mean()),
    ("JI-Last", lambda x: x[-1, 0]),
    ("F1-Last", lambda x: x[-1, 1]),
]

processed = {
    tt: [
        process(tile_data, data, tt, idx, proc=proc, tile_w=TSIZE, tile_h=TSIZE)
        for idx in range(len(data[tt][0]))
    ]
    for tt in tqdm(["train", "test"])
}

# Save
for name, proc in procs:
    processed = {
        tt: [
            process(tile_data, data, tt, idx, proc=proc)
            for idx in range(len(data[tt][0]))
        ]
        for tt in (["train", "test"])
    }
    for tt in ["train", "test"]:
        for idx, im in enumerate(processed[tt]):
            im_ma = masked_equal(im, 0)
            # print(f"tmp_out/tile/{name}-{tt.capitalize()}.{idx:02}.png {im_ma.min()} {im_ma.max()}")
            if im_ma.min() < VMIN:
                print(f"Warning: {tt.capitalize()}.{idx:02} ({im_ma.min()}) < {VMIN}")
            plt.imsave(
                f"tmp_out/tile/{name}-{tt.capitalize()}.{idx:02}.png",
                im_ma,
                cmap="jet",
                vmin=VMIN,
                vmax=1,
            )
arr = np.array(
    [
        [0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75, 0.75],
        [1, 1, 1, 1],
    ]
)
plt.imsave(f"tmp_out/tile/{name}-calibration.png", arr, cmap="jet", vmin=VMIN, vmax=1)
