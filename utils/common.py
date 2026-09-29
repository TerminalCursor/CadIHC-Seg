from matplotlib.colors import Colormap


def preproc_results(results):
    """# Preprocess Dictionary Results

    Flatten dictionary values to a list of values.
    Format floats to 3 decimal places and pass strings through.

    Assume results is in the form of a double dictionary.
    Example: {"a": {"t1": 0, "t2": 2}, "b": {"t1": 0, "t3": 3}}
        becomes 0,2,0,3 (With space paddings so the fields are width-9)
    """
    return ",".join(
        map(
            lambda x: ",".join(
                map(
                    lambda y: f"{y:9}" if isinstance(y, str) else f"{y*100:9.3f}",
                    x.values(),
                )
            ),
            results.values(),
        )
    )


def orig_evaluate_model_m(real_masks, predict_masks):
    """# Model Evaluation on Masks

    real_masks: list of masks
    predict_masks: list of masks
    """
    from cellpose import metrics
    from numpy import uint16
    from .metrics import aggregated_f1

    real_masks = [real_mask.astype(uint16) for real_mask in real_masks]
    predict_masks = [predict_mask.astype(uint16) for predict_mask in predict_masks]
    aji = metrics.aggregated_jaccard_index(real_masks, predict_masks).mean()
    af1 = aggregated_f1(real_masks, predict_masks).mean()
    return {
        "aggregated_jaccard_index": float(aji),
        "aggregated_f1": float(af1),
    }


def orig_evaluate_model(model, data_mask_tuple, diameter=30):
    """# Model Evaluation on Masks

    model: model to evaluate on
    data_mask_tuple: Tuple of the form (images, annotated_masks?, tilers?)
    diameter: Cellpose resize diameter, advised to use a constant 30
    """
    predict_masks = None
    if diameter is None:
        predict_masks, _, _ = model.eval(data_mask_tuple[0])
    else:
        predict_masks, _, _ = model.eval(data_mask_tuple[0], diameter=diameter)
    return evaluate_model_m(data_mask_tuple[1], predict_masks)


def new_evaluate_model_m(real_masks, predict_masks, full=False):
    """# Evaluate Model

    Compute mask metrics: JI, Dice, Precision, Recall, Error.
    A bit inefficient and so this is deemed the slow version.

    real_masks: list of masks
    predict_masks: list of masks
    full: Whether to return the raw tp, fp, and fn
      from Cellposes's average precision compute
    """
    from cellpose import metrics
    from numpy import uint16
    from .metrics import iou
    from .metrics import aggregated_f1

    real_masks = [real_mask.astype(uint16) for real_mask in real_masks]
    predict_masks = [predict_mask.astype(uint16) for predict_mask in predict_masks]
    aji = metrics.aggregated_jaccard_index(real_masks, predict_masks).mean()
    af1 = aggregated_f1(real_masks, predict_masks).mean()
    tp = 0
    fp = 0
    fn = 0
    for real_mask, pred_mask in zip(real_masks, predict_masks):
        _ji, _tp, _fp, _fn, _ = iou(real_mask, pred_mask)
        tp += _tp
        fp += _fp
        fn += _fn
    if full:
        return {
            "aggregated_jaccard_index": float(aji),
            "aggregated_f1": float(af1),
            "jaccard_index": float((tp / (tp + fp + fn))),
            "precision": float((tp / (tp + fp))),
            "recall": float((tp / (tp + fn))),
            "error": float(((fp + fn) / (tp + fn))),
            "f1": float((2 * tp) / (2 * tp + fp + fn)),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return {
        "aggregated_jaccard_index": float(aji),
        "aggregated_f1": float(af1),
        "jaccard_index": float((tp / (tp + fp + fn))),
        "precision": float((tp / (tp + fp))),
        "recall": float((tp / (tp + fn))),
        "error": float(((fp + fn) / (tp + fn))),
        "f1": float((2 * tp) / (2 * tp + fp + fn)),
    }


def new_evaluate_model(model, data_mask_tuple, diameter=30, full=False):
    """# Evaluate Model on Masks

    Compute mask metrics: JI, Dice, Precision, Recall, Error.
    A bit inefficient and so this is deemed the slow version.

    model: model to evaluate on
    data_mask_tuple: Tuple of the form (images, annotated_masks?, tilers?)
    diameter: Cellpose resize diameter, advised to use a constant 30
    """
    predict_masks = None
    if diameter is None:
        predict_masks, _, _ = model.eval(data_mask_tuple[0])
    else:
        predict_masks, _, _ = model.eval(data_mask_tuple[0], diameter=diameter)
    return evaluate_model_m(data_mask_tuple[1], predict_masks, full=full, slow=True)


def evaluate_model_m(real_masks, predict_masks, full=False, slow=False):
    """# Compare Masks across Metrics

    Computes the JI and Dice scores

    real_masks: list of masks
    predict_masks: list of masks
    full: (slow only parameter) computes precision, recall, and error metrics
    slow: Use the inefficient but more comprehensive mode to evaluate
    """
    if slow:
        return new_evaluate_model_m(real_masks, predict_masks, full=full)
    return orig_evaluate_model_m(real_masks, predict_masks)


def evaluate_model(model, data_mask_tuple, diameter=30, full=False, slow=False):
    """# Evaluate Model Performance across Metrics

    Computes the JI and Dice scores of the Model predictions
     with respect to the annotations

    model: model to evaluate on
    data_mask_tuple: Tuple of the form (images, annotated_masks?, tilers?)
    diameter: Cellpose resize diameter, advised to use a constant 30
    full: (slow only parameter) computes precision, recall, and error metrics
    slow: Use the inefficient but more comprehensive mode to evaluate
    """
    if slow:
        return new_evaluate_model(model, data_mask_tuple, diameter=diameter, full=full)
    return orig_evaluate_model(model, data_mask_tuple, diameter=diameter)


def random_colormap(data, base_colormap=None) -> Colormap:
    """# Randomly Color Data

    Instead of having a sequential colormap, randomly color data.
    Useful for coloring masks different colors so adjacent masks are likely distinct.
    Can struggle for large number of unique data points.

    base_colormap: matplotlib.pyplot.cm.jet if `None`
    """
    import numpy as np
    from matplotlib import pyplot as plt

    if base_colormap is None:
        base_colormap = plt.cm.jet
    vals = np.linspace(0, 1, len(np.unique(data)))
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(base_colormap(vals))
    cmap.set_bad(alpha=0)
    return cmap


def logeval(model, logger, train_tuple, test_tuple, time=0, save_images=False):
    """# Log Evaluation Metrics for Model

    model: Model (Cellpose) to evaluate
    logger: utils.logger.Logger to save the data to
    train_tuple: tuple of the form (images, masks, tilers?)
    test_tuple: tuple of the same form
    time: A sequential number to label the images with
    """
    if not logger.enabled:
        return
    from .colors import color_mask
    from matplotlib import pyplot as plt
    import numpy as np

    def fixup(x):
        """# Fixup Array

        Resize Array so it can be easily saved to an image.
        1. Fix dimensions so it is (W,H,3 or None)
        2. Scale data so it fits between 0 and 1
        """

        def _shape_fixup(im):
            """# Fixup Array

            If the array is 3-dimensional:
            If it is depth three, return it untransformed
            Otherwise, assume it is of the form (2, W, H) and return
                a (W,H,3) by padding with zeros
            Otherwise: return it untransformed
            """
            from numpy import dstack, zeros_like

            if len(im.shape) == 3:
                if im.shape[-1] == 3:
                    return im
                return dstack(
                    [im.transpose(1, 2, 0), zeros_like(im[0, ...])[..., None]]
                )
            return im

        # Fix Shape
        y = _shape_fixup(x)
        # Fix Scale
        z = y - y.min()
        w = z / z.max()
        return w

    def _save_model_weights(model, out_path, time, save_images=False):
        """# Save Model Weights

        For the CNN version of Cellpose only
        """
        from matplotlib import pyplot as plt

        if not save_images:
            return
        # Ignore Transformer Model Version
        if model.net.__str__()[0] == "T":
            return
        weights = (
            model.net.downsample.down.res_down_0.conv.conv_0[2]
            .weight.data.clone()
            .detach()
            .cpu()
            .numpy()
        )
        assert weights.shape[0] == 32
        W, H = 6, 6
        out_im = np.zeros((W * (weights.shape[1] + 1), H * (weights.shape[2] + 1), 3))
        for x in range(W):
            for y in range(H):
                index = y * W + x
                if index >= weights.shape[0]:
                    continue
                slcx = slice(
                    (x) * (weights.shape[1] + 1), (x + 1) * (weights.shape[1] + 1) - 1
                )
                slcy = slice(
                    (y) * (weights.shape[1] + 1), (y + 1) * (weights.shape[1] + 1) - 1
                )
                filter_n = weights[index, ...]
                out_im[slcx, slcy, :] = filter_n
        out_im = fixup(out_im)
        plt.imsave(f"{out_path}.{time}.W.png", out_im)

    out_path = logger.filename[: -len(".txt")]
    _save_model_weights(model, out_path, time, save_images=save_images)

    # Individual Evaluation
    ## Training Images
    p_m_f, _, _ = model.eval(train_tuple[0], diameter=30)
    for idx in range(len(train_tuple[0])):
        p_m = p_m_f[idx]
        evaluation_results = {
            "test": {"test": f"T{idx:02}"},
            "results": evaluate_model_m([train_tuple[1][idx]], [p_m]),
        }
        logger.addnl(preproc_results(evaluation_results))

    ## Test Images
    p_m_f, raw_output_f, _ = model.eval(test_tuple[0], diameter=30)
    for idx in range(len(test_tuple[0])):
        p_m = p_m_f[idx]
        raw_output = raw_output_f[idx]
        evaluation_results = {
            "test": {"test": f"E{idx:02}"},
            "results": evaluate_model_m([test_tuple[1][idx]], [p_m]),
        }
        logger.addnl(preproc_results(evaluation_results))

        if save_images:
            # Save the image in the 0th position
            plt.imsave(
                f"{out_path}.{time}.{idx}-0.png", test_tuple[0][idx].transpose(1, 2, 0)
            )
            # Save the Annotated Mask in the 1st position
            plt.imsave(
                f"{out_path}.{time}.{idx}-1.png",
                color_mask(test_tuple[1][idx]),
            )
            # Save the Predicted Mask in the 2nd Position
            plt.imsave(f"{out_path}.{time}.{idx}-2.png", color_mask(p_m))
            # Save the raw outputs to the 3rd+ positions
            #  3 - HSV flow reconstruction
            #  4 - VH flows
            #  5 - Overall Mask
            for jdx in range(len(raw_output)):
                im = fixup(raw_output[jdx])
                plt.imsave(f"{out_path}.{time}.{idx}-{jdx+3}.png", im)

    # Bulk Evaluation
    evaluation_results = {
        "train_set": evaluate_model(model, train_tuple),
        "test_set": evaluate_model(model, test_tuple),
    }
    logger.addnl(preproc_results(evaluation_results))


def get_tiles(data_tuple, data_index, n_tiles=8, randomize=False):
    """# Get a Tile

    data_tuple: Tuple of the form (images, annotations, tilers)
    data_index: Image to get the tiles from
    n_tiles: Number of tiles to return, all if `None`
    randomize: Randomize which tiles to poll
    """
    from numpy.random import choice
    from .tile import point2slice, sliceNimNmask2tuple

    images, masks, tilers = data_tuple
    tile_points = tilers[data_index].points.copy()
    if randomize:
        tile_points = tile_points[
            choice(tile_points.shape[0], tile_points.shape[0], replace=False), ...
        ]
    if n_tiles is not None:
        tile_points = tile_points[:n_tiles, ...]
    tile_slices = [
        point2slice(tilers[data_index], tile_point) for tile_point in tile_points
    ]
    image = images[data_index]
    mask = masks[data_index]
    tiles = [sliceNimNmask2tuple(tile_slice, image, mask) for tile_slice in tile_slices]
    return tiles


def logtile(
    model, logger, train_tuple, test_tuple, time=0, n_tiles=None, randomize=False
):
    """# Log Evaluation Metrics for Model across Tiles

    model: Model (Cellpose) to evaluate
    logger: utils.logger.Logger to save the data to
    train_tuple: tuple of the form (images, masks, tilers)
    test_tuple: tuple of the same form
    time: A sequential number to label the images with
    n_tiles: Number of tiles in each image to log, all if `None`
    randomize: Randomize tiles to poll in each image
    """
    if not logger.enabled:
        return
    ## Training Images
    for idx in range(len(train_tuple[0])):
        tiles = get_tiles(train_tuple, idx, n_tiles=n_tiles, randomize=randomize)
        for jdx in range(len(tiles)):
            tup = ([tiles[jdx][0]], [tiles[jdx][1]])
            ev = evaluate_model(model, tup)
            evaluation_results = {
                "test": {"test": f"T{idx:02}.t{jdx:03}"},
                "results": ev,
            }
            logger.addnl(preproc_results(evaluation_results))
            # Log tiler Result
            train_tuple[2][idx].eval(jdx, *list(ev.values()))

    ## Test Images
    for idx in range(len(test_tuple[0])):
        tiles = get_tiles(test_tuple, idx, n_tiles=n_tiles, randomize=randomize)
        for jdx in range(len(tiles)):
            tup = ([tiles[jdx][0]], [tiles[jdx][1]])
            ev = evaluate_model(model, tup)
            evaluation_results = {
                "test": {"test": f"E{idx:02}.t{jdx:03}"},
                "results": ev,
            }
            logger.addnl(preproc_results(evaluation_results))
            # Log tiler Result
            test_tuple[2][idx].eval(jdx, *list(ev.values()))


def load_image(ffile, halve=True):
    """# Load Image

    1. Load an image
    2. Move axes to (channel, width, height)
    3. Transform it under standard normalization transformation
    4. Subsample by a factor of two if halve is true
    """
    from matplotlib import pyplot as plt
    from .image import transform_image
    image = plt.imread(ffile)
    assert len(image.shape) == 3
    if image.shape[0] > 4:
        image = image.transpose(2, 0, 1)
    if image.shape[0] == 4:
        image = image[:3, ...]
    image = transform_image(image)
    if halve:
        image = image[:, ::2, ::2]
    return image


def load_pth(pth):
    """# Load a Cellpose pth"""
    from . import ModelBuildConfig
    from cellpose import models
    cfg = ModelBuildConfig()
    cfg.pretrained_model = pth
    return models.CellposeModel(**cfg.todict())


def segment(image, c_mstr, n_mstr, restrict=True, closeup=True, masked=False):
    """# Segment an Image

    c_mstr: Cell-segmentation model pth string
    n_mstr: Nucleus-segmentation model pth string (None = no nucleus processing)
    restrict: Ignore nuclei outside cell-positive region
    closeup: Whether to closeup stained regions
    masked: Whether to return a masked numpy array for segmentations

    Returns masks and raw evaluation outputs"""
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

    c_model = load_pth(c_mstr)

    n_model = None
    if n_mstr is not None:
        n_model = load_pth(n_mstr)

    # Cellpose Segmentation
    c_ma = None
    ma, c_raw, _ = c_model.eval([image])
    c_raw = c_raw[0][2]
    if closeup:
        # Try to closeup holes
        try:
            c_ma = postforward(c_model, image)
        # Otherwise just use original mask
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
