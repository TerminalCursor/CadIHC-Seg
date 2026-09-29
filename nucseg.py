#!/usr/bin/env python
# coding: utf-8
from utils import *
from numpy import uint8
from numpy.ma import masked_equal
from matplotlib import pyplot as plt
from cellpose import models
from tqdm.auto import tqdm
from time import perf_counter

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
    return image[:, ::2, ::2]


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
        y = float(y) / mav.shape[1]
        data.append((x, y))
    return data, oim


def outline(mask, width=2, background=0, masked=True):
    internal = mask.copy()
    try:
        internal = internal.data
    except:
        pass
    new_mask = internal.copy()
    assert width > 0
    for _ in range(width):
        internal = erode_all(internal, background=background)
    # Remove Internal Regions
    internal_mask = internal > 0
    new_mask[internal_mask] = 0
    if masked:
        new_mask = np.ma.masked_equal(new_mask, 0)
    return new_mask


def _process(fname):
    t0 = perf_counter()
    im = load_image(fname)
    obase = fname[: -len(".jpg")]
    c_ma, n_ma, c_r, n_r = segment(
        im, "promising/Model-C00.pt", "promising/Model-N00.pt", masked=True
    )
    t1 = perf_counter()
    W = 4
    c_out = outline(c_ma, width=W)
    n_out = outline(n_ma, width=W)
    c2_out = outline(c_ma, width=2 * W)
    n2_out = outline(n_ma, width=2 * W)
    t2 = perf_counter()
    (cd, nd), (cim, nim) = dot_mask([c_ma, n_ma], im=im)
    t3 = perf_counter()
    plt.imsave(f"{obase}_Cell.png", color_mask(c_ma))
    plt.imsave(f"{obase}_CellMask.png", c_r)
    plt.imsave(f"{obase}_CellDot.png", cim.transpose(1, 2, 0))
    plt.imsave(f"{obase}_Nuclei.png", color_mask(n_ma), cmap="cool")
    plt.imsave(f"{obase}_NucleiRegular.png", color_mask(n_ma))
    plt.imsave(f"{obase}_NucleiMask.png", n_r)
    plt.imsave(f"{obase}_NucleiDot.png", nim.transpose(1, 2, 0))
    plt.imsave(f"{obase}_CellOutline.png", color_mask(c_out))
    plt.imsave(f"{obase}_NucleiOutline.png", color_mask(n_out))
    t4 = perf_counter()
    np.savetxt(f"{obase}_Cell.csv", cd)
    np.savetxt(f"{obase}_Nuclei.csv", nd)
    t5 = perf_counter()
    c_data, c_maps, _, c_v = full_energetics(
        "promising/Model-C00.pt", "promising/Model-N00.pt", im, _width=2, fixup=True
    )
    for idx in range(3):
        _data = c_v[0][idx,...]
        _mask = _data > 0
        print(_data[_mask].min())
        print(_data[_mask].max())
    _data = c_v[0][3,...]
    _mask = _data > 0
    print(_data[_mask].min())
    print(_data[_mask].max())
    t6 = perf_counter()
    ## from numpy.ma import masked_array

    ## assert isinstance(c_ma, MaskedArray)
    ## cell_mask = np.ma.getdata(c_ma) == 0
    ## c1___mask = np.ma.getdata(c_out) == 0
    ## c2___mask = np.ma.getdata(c2_out) == 0
    ## pressure_energy = color_mask(np.ma.getdata(c_maps[0][..., 0]), None)
    ## pressure_energy = masked_array(pressure_energy, cell_mask)
    ## cc_adhesion_energy = color_mask(np.ma.getdata(c_maps[0][..., 2]), None)
    ## cc_adhesion_energy = masked_array(cc_adhesion_energy, c2___mask)
    ## ct_adhesion_energy = color_mask(np.ma.getdata(c_maps[0][..., 3]), None)
    ## ct_adhesion_energy = masked_array(ct_adhesion_energy, c1___mask)
    ## plt.imsave(f"{obase}_Pressure.png", pressure_energy, cmap="Blues")
    ## plt.imsave(f"{obase}_CCAdhesion.png", cc_adhesion_energy, cmap="Reds")
    ## plt.imsave(f"{obase}_CTAdhesion.png", ct_adhesion_energy, cmap="Greens")
    plt.imsave(f"{obase}_Energy.png", c_v[0])
    t7 = perf_counter()
    times = np.array([t1, t2, t3, t4, t5, t6, t7]) - np.array(
        [t0, t1, t2, t3, t4, t5, t6]
    )
    print(times)

    c_data, c_maps, _, c_v = full_energetics(
        "promising/Model-C00.pt", "promising/Model-N00.pt", im, _width=1, fixup=True
    )
    ## from numpy.ma import masked_array

    ## assert isinstance(c_ma, MaskedArray)
    ## cell_mask = np.ma.getdata(c_ma) == 0
    ## c1___mask = np.ma.getdata(c_out) == 0
    ## c2___mask = np.ma.getdata(c2_out) == 0
    ## pressure_energy = color_mask(np.ma.getdata(c_maps[0][..., 0]), None)
    ## pressure_energy = masked_array(pressure_energy, cell_mask)
    ## cc_adhesion_energy = color_mask(np.ma.getdata(c_maps[0][..., 2]), None)
    ## cc_adhesion_energy = masked_array(cc_adhesion_energy, c2___mask)
    ## ct_adhesion_energy = color_mask(np.ma.getdata(c_maps[0][..., 3]), None)
    ## ct_adhesion_energy = masked_array(ct_adhesion_energy, c1___mask)
    ## plt.imsave(f"{obase}_ThinPressure.png", pressure_energy, cmap="Blues")
    ## plt.imsave(f"{obase}_ThinCCAdhesion.png", cc_adhesion_energy, cmap="Reds")
    ## plt.imsave(f"{obase}_ThinCTAdhesion.png", ct_adhesion_energy, cmap="Greens")
    plt.imsave(f"{obase}_ThinEnergy.png", c_v[0])

    W, H = (96 * 2, 48 * 2)
    ri = np.random.randint
    x, y = ri(c_ma.shape[0] - W), ri(c_ma.shape[1] - H)
    assert c_ma.shape[0] == im.shape[1]
    assert c_ma.shape[1] == im.shape[2]
    plt.imsave(f"{obase}_CR_Im.png", im.transpose(1, 2, 0)[x : x + W, y : y + H, :])
    plt.imsave(f"{obase}_CR_Cm.png", color_mask(c_ma[x : x + W, y : y + H]))
    plt.imsave(f"{obase}_CR_Nm.png", color_mask(n_ma[x : x + W, y : y + H]))
    plt.imsave(
        f"{obase}_CR_Nma.png", color_mask(n_ma[x : x + W, y : y + H]), cmap="cool"
    )
    plt.imsave(f"{obase}_CR_Co.png", color_mask(c_out[x : x + W, y : y + H]))
    plt.imsave(
        f"{obase}_CR_Noa.png", color_mask(n_out[x : x + W, y : y + H]), cmap="cool"
    )
    #plt.imsave(
    #    f"{obase}_CR_Pressure.png", pressure_energy[x : x + W, y : y + H], cmap="Blues"
    #)
    #plt.imsave(
    #    f"{obase}_CR_CCAdhesion.png",
    #    cc_adhesion_energy[x : x + W, y : y + H],
    #    cmap="Reds",
    #)
    #plt.imsave(
    #    f"{obase}_CR_CTAdhesion.png",
    #    ct_adhesion_energy[x : x + W, y : y + H],
    #    cmap="Greens",
    #)


base = "../../home/Downloads/Images/"
from glob import glob

for fname in glob(base + "C7.jpg"):
    _process(fname)
    assert False

for fname in glob(base + "*.jpg"):
    _process(fname)
    # assert False

"""
ex_im = load_image("../BaseImage.png")

c_ma, n_ma, c_r, n_r = segment(ex_im, "promising/Model-C00.pt", "promising/Model-N00.pt", masked=True)

plt.imsave("tmp_out/S-CR.png", c_r)
plt.imsave("tmp_out/S-I.png", ex_im.transpose(1,2,0))
plt.imsave("tmp_out/S-C.png", color_mask(c_ma))
plt.imsave("tmp_out/S-N.png", color_mask(n_ma))

(cd,nd), (cim,nim) = dot_mask([c_ma, n_ma], im=ex_im)
plt.imsave("tmp_out/S-Cell.png", cim.transpose(1,2,0))
plt.imsave("tmp_out/S-Nuclei.png", nim.transpose(1,2,0))
np.savetxt("tmp_out/S-Cell.csv", cd)
np.savetxt("tmp_out/S-Nuclei.csv", nd)
"""
