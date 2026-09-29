from numpy import uint32, uint16

DTYPE = uint16


def my_flood_fill(source_mask, background=0):
    """# Flood Fill A Mask"""
    # Some of these masks are quite large and so the recursion goes deep
    import sys

    sys.setrecursionlimit(3000)
    from numpy import zeros_like

    # If you give it a list of masks, return a flood fill for each
    if isinstance(source_mask, list):
        return [flood_fill(src_mask, background=background) for src_mask in source_mask]
    # Otherwise assume a 2d-mask
    assert len(source_mask.shape) == 2, "Only 2D Arrays Supported"
    flood_filled_mask = zeros_like(source_mask, DTYPE)

    def fill(src, dst, x, y, dst_color, src_color=True, background=0):
        """# Fill A Region

        Color a region with dst_color where the src is the src_color"""
        # Ensure we are still within the bounds of the 2d array
        if x < 0:
            return
        if x >= src.shape[0]:
            return
        if y < 0:
            return
        if y >= src.shape[1]:
            return
        # Don't paint over any colors that aren't the source color
        if src[x, y] != src_color:
            return
        # Don't paint over regions already painted
        if dst[x, y] != background:
            return
        # Paint
        dst[x, y] = dst_color
        # Fill adjacent regions
        fill(
            src,
            dst,
            x - 1,
            y,
            dst_color=dst_color,
            src_color=src_color,
            background=background,
        )
        fill(
            src,
            dst,
            x + 1,
            y,
            dst_color=dst_color,
            src_color=src_color,
            background=background,
        )
        fill(
            src,
            dst,
            x,
            y - 1,
            dst_color=dst_color,
            src_color=src_color,
            background=background,
        )
        fill(
            src,
            dst,
            x,
            y + 1,
            dst_color=dst_color,
            src_color=src_color,
            background=background,
        )

    starting_color = 1
    current_color = starting_color
    for x in range(source_mask.shape[0]):
        for y in range(source_mask.shape[1]):
            if source_mask[x, y] == background:
                continue
            fill(
                source_mask,
                flood_filled_mask,
                x,
                y,
                dst_color=current_color,
                src_color=source_mask[x, y],
                background=background,
            )
            if flood_filled_mask[x, y] == current_color:
                current_color += 1
    return flood_filled_mask


def flood_fill(source_mask, background=0):
    from numpy import zeros_like

    try:
        from skimage.segmentation import flood as skff

        assert background == 0
        dest_mask = zeros_like(source_mask, DTYPE)
        current_color = 1
        for x in range(source_mask.shape[0]):
            for y in range(source_mask.shape[1]):
                if source_mask[x, y] == background:
                    continue
                if dest_mask[x, y] != 0:
                    continue
                filled = skff(source_mask.copy().astype(DTYPE), (x, y))
                dest_mask[filled] = current_color
                current_color += 1
        return dest_mask
    except ImportError as e:
        return my_flood_fill(source_mask, background=0)


def bound_data(mask, label, *rest, nan_background=False):
    """# Restrict 2d- arrays to region of interest
    mask: 2d array
    label: label in mask
    rest: arrays that have their last 2 dimensions the same as the mask
    """
    assert len(mask.shape) == 2
    assert (mask == label).any()
    from numpy import where, single, double, nan

    r, c = where(mask == label)
    xl, xh = r.min(), r.max() + 1
    yl, yh = c.min(), c.max() + 1
    return_arrays = []
    return_arrays.append(mask[xl:xh, yl:yh])
    for arr in rest:
        carr = arr.copy()
        if carr.dtype in [single, double]:
            carr[..., mask != label] = nan if nan_background else arr.min()
        else:
            carr[..., mask != label] = 0
        return_arrays.append(carr[..., xl:xh, yl:yh])
    return return_arrays


def determine_fixup_regions(model, image):
    """# Determine Regions Needing Fixing

    model: Cellpose Model
    image: Image(s) of interest"""
    from numpy import array, unique

    if isinstance(image, list):
        return [determine_fixup_regions(model, img) for img in image]
    assert len(image.shape) == 3
    mask, raw, _ = model.eval([image])
    mask = mask[0]
    raw = raw[0]
    vh = raw[1]
    cell_p = raw[2]
    fixup_region = (cell_p > 0) & (mask == 0)
    return fixup_region, mask, vh, cell_p


def fixup_mask(model, image, nan_background=False):
    from cellpose.dynamics import compute_masks
    from numpy import array, unique

    fixup_regions, mask, vh, cp = determine_fixup_regions(model, image)
    fixup_labels = flood_fill(fixup_regions, background=False)
    label_occurrence = array(unique(fixup_labels, return_counts=True)).T
    fixed = mask.copy()
    for ex_label, freq in label_occurrence:
        if ex_label == 0:
            continue
        bounded_outputs = bound_data(
            fixup_labels, ex_label, mask, vh, cp, nan_background=nan_background
        )
        new_pred = bounded_outputs[0]
        if freq > 10:
            new_pred = compute_masks(
                bounded_outputs[-2], bounded_outputs[-1], max_size_fraction=1
            )
        if len(unique(new_pred)) == 1:
            new_pred = bounded_outputs[0]
        restricted = new_pred[bounded_outputs[0] == ex_label]
        new_res = restricted.copy() + fixed.max()
        fixed[fixup_labels.copy() == ex_label] = new_res

    return flood_fill(fixed)


def _neq_neighbors(mask, x, y):
    def _check_neq(mask, x, y, dx, dy):
        if ((x + dx) < 0) or ((x + dx) >= mask.shape[0]):
            return True
        if ((y + dy) < 0) or ((y + dy) >= mask.shape[1]):
            return True
        return mask[x + dx, y + dy] != mask[x, y]

    lr = [_check_neq(mask, x, y, dx, 0) for dx in [-1, 1]]
    ud = [_check_neq(mask, x, y, 0, dy) for dy in [-1, 1]]
    lurd = [lr[0], ud[0], lr[1], ud[1]]
    lu = lurd[0] and lurd[1]
    ur = lurd[1] and lurd[2]
    rd = lurd[2] and lurd[3]
    dl = lurd[3] and lurd[0]
    return lu or ur or rd or dl


def _neighbors(mask, x, y):
    nbrs = set()
    if x > 0:
        nbrs |= {
            int(mask[x - 1, y]),
        }
    if (x + 1) < mask.shape[0]:
        nbrs |= {
            int(mask[x + 1, y]),
        }
    if y > 0:
        nbrs |= {
            int(mask[x, y - 1]),
        }
    if (y + 1) < mask.shape[1]:
        nbrs |= {
            int(mask[x, y + 1]),
        }
    return nbrs - {
        int(mask[x, y]),
    }


def dilate(mask, background=0):
    new_mask = mask.copy()
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            c = mask[x, y]
            if c != background:
                continue
            nbrs = _neighbors(mask, x, y)
            if len(nbrs) > 0:
                nc = list(nbrs)[0]
                new_mask[x, y] = nc
    return new_mask


def erode_all_old(mask, background=0):
    """
    This implementation can split a mask into two
     since it only checks for a single neighbor
    """
    new_mask = mask.copy()
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            c = mask[x, y]
            if c == background:
                continue
            nbrs = _neighbors(mask, x, y)
            if len(nbrs) > 0:
                new_mask[x, y] = background
    return new_mask


def erode_all_new(mask, background=0):
    """
    This implementation will never split a mask
     as it checks if it is a corner/ pixel-wide protrusion only
    """
    new_mask = mask.copy()
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            c = mask[x, y]
            if c == background:
                continue
            if _neq_neighbors(mask, x, y):
                new_mask[x, y] = background
    return new_mask


def erode_all(mask, background=0, old=False):
    if old:
        return erode_all_old(mask, background=background)
    return erode_all_new(mask, background=background)


def erode(mask, background=0):
    new_mask = mask.copy()
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            c = mask[x, y]
            if c == background:
                continue
            nbrs = _neighbors(mask, x, y)
            if background in nbrs:
                new_mask[x, y] = background
    return new_mask


def close_all(mask, background=0, n=2, old=False):
    background_m = mask.copy() == 0
    ret_mask = mask.copy()
    for _ in range(n):
        ret_mask = erode_all(ret_mask, background=background, old=old)
    for _ in range(n):
        ret_mask = dilate(ret_mask, background=background)
    # Absorb removed smaller masks into larger masks
    for _ in range(n):
        ret_mask = dilate(ret_mask, background=background)
    # Restore original background
    ret_mask[background_m] = background
    return ret_mask


def postforward(model, image, n=4, nan_background=False, old=False):
    from cellpose.dynamics import dynamics_logger

    dynamics_logger.setLevel("ERROR")
    # If given a list of images, return for each image
    if isinstance(image, list) or isinstance(image, tuple):
        return [
            postforward(model, i, n=n, nan_background=nan_background) for i in image
        ]
    # Compute Mask Including Missing Regions
    fix = fixup_mask(model, image, nan_background=nan_background)
    # Return Masks Above a Certain Width
    return close_all(fix, n=n, old=old)
