#!/usr/bin/env python
# coding: utf-8
from utils import *
from numpy.ma import masked_equal
from matplotlib import pyplot as plt
from cellpose import models
from tqdm.auto import tqdm
from skimage.measure import regionprops_table
import pandas as pd

config = UnifiedConfig()

train_data, test_data = prepare_data(config.dataset.base_directory)

ex_im = test_data[0][0]


def load_image(ffile):
    image = plt.imread(ffile)
    assert len(image.shape) == 3
    if image.shape[0] > 4:
        image = image.transpose(2, 0, 1)
    if image.shape[0] == 4:
        image = image[:3, ...]
    image = transform_image(image)
    image = image[:, ::2, ::2]
    return image


def segment(image, c_mstr, n_mstr, restrict=True, closeup=True, masked=False):
    from numpy import unique, zeros_like

    # Returns the cellpose segmentation and the nuclear segmentation
    if isinstance(image, list):
        results = [
            segment(
                im, c_mstr, n_mstr, restrict=restrict, closeup=closeup, masked=masked
            )
            for im in image
        ]
        return zip(*results)

    c_cfg = ModelBuildConfig()
    c_cfg.pretrained_model = c_mstr
    c_model = models.CellposeModel(**c_cfg.todict())

    n_cfg = None
    n_model = None
    if n_mstr is not None:
        n_cfg = ModelBuildConfig()
        n_cfg.pretrained_model = n_mstr
        n_model = models.CellposeModel(**n_cfg.todict())

    # Cellpose Segmentation
    c_ma = None
    ma, c_raw, _ = c_model.eval([image])
    c_raw = c_raw[0][2]
    if closeup:
        try:
            c_ma = postforward(c_model, image)
        except:
            c_ma = ma[0]
    else:
        c_ma = ma[0]

    n_ma = None
    n_raw = None
    if n_model is not None:
        ma, n_raw, _ = n_model.eval([image])
        n_raw = n_raw[0][2]
        n_ma_raw = ma[0]

        if restrict:
            n_ma = zeros_like(n_ma_raw)

            c_mask = c_ma > 0
            n_overlap = n_ma_raw[c_mask]
            n_keep = list(
                set([int(v) for v in unique(n_overlap)])
                - {
                    0,
                }
            )
            detection_number = 1
            for keep in n_keep:
                n_ma[n_ma_raw == keep] = detection_number
                detection_number += 1
        else:
            n_ma = n_ma_raw

    if masked:
        from numpy.ma import masked_equal

        c_ma = masked_equal(c_ma, 0)
        if n_ma is not None:
            n_ma = masked_equal(n_ma, 0)
    return c_ma, n_ma, c_raw, n_raw


def dot_mask(ma, im=None, w=4):
    if isinstance(ma, list):
        res = [dot_mask(m, im=im, w=w) for m in ma]
        return zip(*res)
    from scipy.ndimage import center_of_mass
    from numpy import unique
    from numpy.ma import MaskedArray

    if isinstance(ma, MaskedArray):
        mav = ma.data
    else:
        mav = ma
    oim = None
    if im is not None:
        oim = im.copy()
    mask_vs = list(
        set([int(v) for v in unique(mav)])
        - {
            0,
        }
    )
    data = []
    D = w // 2
    d = w % 2
    for v in mask_vs:
        x, y = center_of_mass(mav == v)
        if oim is not None:
            X = int(x)
            Y = int(y)
            xs = slice(max(0, X - D), X + D + d)
            ys = slice(max(0, Y - D), Y + D + d)
            oim[0, xs, ys] = 1
            oim[1:, xs, ys] = 0
        x = float(x) / mav.shape[0]
        y = float(x) / mav.shape[1]
        data.append((x, y))
    return data, oim


def _process(fname, thresh=30):
    from numpy import unique, floor, array, hstack, zeros
    from numpy.ma import masked
    from json import dumps
    import time
    from pandas import read_csv, Categorical

    time_i = time.perf_counter()
    cdataf = fname + "CellCellMatches.npy"
    ndataf = fname + "CellNucleusMatches.npy"
    cdata = np.load(cdataf, allow_pickle=True)
    ndata = np.load(ndataf, allow_pickle=True)
    data = hstack([cdata, ndata])
    link_data = data[:, [1, 3]]
    _filter = link_data == None
    _filter = _filter.any(axis=1)
    found_links = link_data[~_filter]
    # Find all unique cells
    found_links_c = found_links[:, 0]
    _, first_idx = unique(found_links_c, return_index=True)
    mask = zeros(len(found_links_c), dtype=bool)
    mask[first_idx] = True
    found_links = found_links[mask, :]
    # Find all unique nuclei
    found_links_n = found_links[:, 1]
    _, first_idx = unique(found_links_n, return_index=True)
    mask = zeros(len(found_links_n), dtype=bool)
    mask[first_idx] = True
    found_links = found_links[mask, :]
    # Done
    cdataf = fname + "Metrics.csv"
    ndataf = fname + "NucleusRegionProps.csv"
    df_c = read_csv(cdataf)
    df_n = read_csv(ndataf)
    df_co = df_c.copy()
    df_co = df_co[df_co["CellNum"].isin(found_links[:, 0])]
    df_co_f = df_co.copy()
    df_co_f["CellNum"] = Categorical(
        df_co["CellNum"],
        categories=found_links[:, 0],
        ordered=True,
    )
    df_co = df_co_f.sort_values("CellNum")
    df_co = df_co.reset_index(drop=True)
    df_no = df_n.copy()
    df_no = df_no[df_no["label"].isin(found_links[:, 1])]
    df_no_f = df_no.copy()
    df_no_f["label"] = Categorical(
        df_no["label"],
        categories=found_links[:, 1],
        ordered=True,
    )
    df_no = df_no_f.sort_values("label")
    df_no = df_no.reset_index(drop=True)
    df = df_co.join(df_no)
    df.to_csv(fname + "CNJoinedMetrics.csv")
    time_f = time.perf_counter()
    print(time_f - time_i)
    return df


base = "../../home/Downloads/Images/"
base = "../Datasets/*/image.png"
base = "../Datasets/*/"
from glob import glob

for fname in glob(base):
    try:
        _process(fname)
    except Exception as e:
        print(f"Err on {fname}")
        print(e)
