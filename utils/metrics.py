def iou(real, pred, threshold=0.5, pixel_level=False):
    from numpy import unique, ndarray, zeros
    from itertools import product
    from tqdm.auto import tqdm

    # tqdm = lambda x: x
    assert isinstance(real, ndarray)
    assert real.shape == pred.shape
    assert len(real.shape) == 2
    real_v = list(
        set(unique(real).tolist())
        - {
            0,
        }
    )
    pred_v = list(
        set(unique(pred).tolist())
        - {
            0,
        }
    )
    scores = zeros((len(real_v), len(pred_v), 4))
    for r_vi in tqdm(range(len(real_v))):
        r_v = real_v[r_vi]
        r_m = real == r_v
        # Overlapping Masked Region on Prediction Mask
        o_m = set(unique(pred[r_m]).tolist()) - {
            0,
        }
        # Get the indices of the values
        o_v = [pred_v.index(value) for value in o_m]
        for p_vi in tqdm(o_v, leave=False):
            p_v = pred_v[p_vi]
            p_m = pred == p_v
            tp = (r_m & p_m).sum()
            if tp == 0:
                fp = p_m.sum()
                fn = r_m.sum()
            else:
                d_m = r_m.astype(int) - p_m.astype(int)
                fp = (d_m == -1).sum()
                fn = (d_m == 1).sum()
            scores[r_vi, p_vi, 0] = tp / (tp + fp + fn)
            scores[r_vi, p_vi, 1] = tp
            scores[r_vi, p_vi, 2] = fp
            scores[r_vi, p_vi, 3] = fn
            if (tp / (tp + fp + fn)) < 0.5:
                continue
            break
    TP = 0
    FP = 0
    FN = 0
    used_r = set()
    used_p = set()
    map_rgb = None
    if pixel_level:
        map_rgb = zeros((real.shape[0], real.shape[1], 3), dtype=bool)
    # Red = Real mask but no prediction
    # Green = Real mask and prediction
    # Cyan (Green + Blue) = Prediction with no real mask
    # Yellow = Prediction with no real mask
    # map_rgb[...,0] = (real > 0)
    # map_rgb[...,1] = (pred > 0)
    for r_vi, r_v in enumerate(real_v):
        sub_array = scores[r_vi, :, :]
        p_vi = sub_array[..., 0].argmax()
        if sub_array[p_vi, 0] < 0.5:
            continue
        if not pixel_level:
            TP += 1
            continue
        r_m = real == r_v
        p_m = pred == pred_v[p_vi]
        map_rgb[..., 1] |= p_m
        map_rgb[..., 0] |= (r_m) & (~p_m)
        map_rgb[..., 2] |= (~r_m) & (p_m)
        TP += sub_array[p_vi, 1]
        FP += sub_array[p_vi, 2]
        FN += sub_array[p_vi, 3]
        used_r |= {r_vi}
        used_p |= {p_vi}
    # print(TP/(TP + FP + FN), TP / (TP + FP), TP / (TP + FN), 2*TP/(2*TP + FP + FN))
    unused_r = set(range(len(real_v))) - used_r
    unused_p = set(range(len(pred_v))) - used_p
    for r_vi in unused_r:
        if not pixel_level:
            FN += 1
            continue
        FN += (real == real_v[r_vi]).sum()
        r_m = real == r_v
        # map_rgb[...,0] |= r_m
    for p_vi in unused_p:
        if not pixel_level:
            FP += 1
            continue
        FP += (pred == pred_v[p_vi]).sum()
        p_m = pred == pred_v[p_vi]
        # map_rgb[...,2] |= p_m
    # return TP/(TP + FP + FN), TP / (TP + FP), TP / (TP + FN), 2*TP/(2*TP + FP + FN)
    # return float(TP/(TP + FP + FN)), float(TP), float(FP), float(FN)
    return float(TP / (TP + FP + FN)), int(TP), int(FP), int(FN), map_rgb


def iou_eval_tuple(model, dm_tuple):
    real_mask = dm_tuple[1]
    pred_mask, _, _ = model.eval([dm_tuple[0]])
    pred_mask = pred_mask[0]
    return iou(real_mask, pred_mask)


def iou_eval_model(model, dm_tuples):
    iou_data = [
        iou_eval_tuple(model, (dm_tuples[0][idx], dm_tuples[1][idx]))
        for idx in range(len(dm_tuples[0]))
    ]
    tp = 0
    fp = 0
    fn = 0
    for _, _tp, _fp, _fn, _ in iou_data:
        tp += _tp
        fp += _fp
        fn += _fn
    return (
        tp / (tp + fp + fn),
        tp,
        fp,
        fn,
        tp / (tp + fp),
        tp / (tp + fn),
        2 * tp / (2 * tp + fp + fn),
    )


def aggregated_f1(masks_true, masks_pred):
    from cellpose.metrics import mask_ious, _label_overlap
    from numpy import arange, logical_or, zeros

    """
    AF1 = intersection of all matched masks / union of all masks

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)):
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)):
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        af1 (float): aggregated jaccard index for each set of masks
    """
    af1 = zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = logical_or(masks_true[n] > 0, masks_pred[n] > 0).sum()
        overlap = overlap[inds[preds > 0] + 1, preds[preds > 0].astype(int)]
        af1[n] = (2 * overlap.sum()) / (overlap.sum() + union)
    return af1
