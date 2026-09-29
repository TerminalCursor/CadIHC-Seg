def rescale(data, _min=128, _max=255):
    """
    Rescale a 2D uint8 NumPy array so that:
    - Zero values remain zero
    - The smallest non-zero value becomes 64
    - The maximum value becomes _max

    Parameters
    ----------
    data : np.ndarray
        2D uint8 array

    Returns
    -------
    np.ndarray
        Rescaled uint8 array
    """
    from numpy import uint8, float32, log10
    from numpy import any as npany
    from numpy import round as npround

    if data.dtype != uint8:
        raise ValueError("Input must be uint8")
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array")

    result = data.copy().astype(float32)

    # Mask of non-zero values
    mask = result > 0
    if not npany(mask):
        print("Nothing found!")
        return result  # all zeros

    # result[mask] = log10(result[mask]+2)
    nonzero_vals = result[mask]
    vmin = nonzero_vals.min()
    vmax = nonzero_vals.max()

    if vmin == vmax:
        # All non-zero values are identical → set them to _max
        print("All equal")
        result[mask] = _max
        return result

    # Linear rescaling to [_min, _max]
    scaled = _min + (nonzero_vals - vmin) * (_max - _min) / (vmax - vmin)
    result[mask] = npround(scaled).astype(uint8)

    return result


def expand(arr):
    """
    Expands all non-zero regions in a 2D uint16 array outward by one pixel
    (8-connected neighborhood), preserving original values.

    New pixels take the maximum value of neighboring non-zero pixels.

    Parameters
    ----------
    arr : np.ndarray
        2D array of dtype uint16

    Returns
    -------
    np.ndarray
        2D uint16 array with expanded regions
    """
    from numpy import uint16, stack, max, pad

    if arr.dtype != uint16:
        raise ValueError("Input array must have dtype uint16")
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array")

    h, w = arr.shape

    # Pad array to simplify boundary handling
    padded = pad(arr, pad_width=1, mode="constant", constant_values=0)

    # Collect all 3x3 neighborhood shifts
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbors.append(padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])

    # Stack and compute pixelwise maximum across neighborhood
    stacked = stack(neighbors, axis=0)
    expanded = max(stacked, axis=0)

    return expanded.astype(uint16)


def logical_or(li):
    """# Extend Logical Or To Multiple Lists

    logical_or([A, B, C, ...]) = A or B or C or ... for each element
    """
    from numpy import logical_or as _logical_or

    if not isinstance(li, list):
        li = list(li)
    if len(li) == 2:
        return _logical_or(li[0], li[1])
    return _logical_or(li[0], logical_or(li[1:]))


def compute_energetics(mask, _width=2):
    """# Compute the Energetics of a Mask

    mask: (W,H) of integer type

    returns: (W,H,4)
    Where the 4 channels are the pressure, tension, inter-mask adhesion, mask-background adhesion
    """
    if isinstance(mask, list):
        return [compute_energetics(m) for m in mask]
    from scipy.signal import convolve2d
    from numpy import uint8, uint16, unique, zeros
    from numpy.ma import masked_equal, masked_array, getdata

    BACKGROUND = 0
    kernels = {
        "up": (
            [
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 0],
            ],
            slice(0, -2),
            slice(1, -1),
        ),
        "down": (
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
            ],
            slice(2, None),
            slice(1, -1),
        ),
        "left": (
            [
                [0, 0, 0],
                [1, -1, 0],
                [0, 0, 0],
            ],
            slice(1, -1),
            slice(0, -2),
        ),
        "right": (
            [
                [0, 0, 0],
                [0, -1, 1],
                [0, 0, 0],
            ],
            slice(1, -1),
            slice(2, None),
        ),
    }

    cell_kernels = {
        direction: ((convolve2d(mask, kernel)[sl1, sl2] != 0) & (mask != 0)).astype(
            uint8
        )
        for direction, (kernel, sl1, sl2) in kernels.items()
    }
    binary_kernels = {
        direction: ((convolve2d(mask > 0, kernel)[sl1, sl2] != 0) & (mask != 0)).astype(
            uint8
        )
        for direction, (kernel, sl1, sl2) in kernels.items()
    }

    # Boundary Maps
    # Cell-Tissue: Only care about the boundary between zero and non-zero regions
    cell_tissue_maps = {
        direction: (binary_kernels[direction]).astype(uint8)
        for direction in kernels.keys()
    }
    cell_tissue_map = logical_or(cell_tissue_maps.values()).astype(uint8)
    # Cell-Cell: Care about the cell-cell regions and ignore the cell-background
    cell_cell_maps = {
        direction: (cell_kernels[direction] & (~binary_kernels[direction])).astype(
            uint8
        )
        for direction in kernels.keys()
    }
    cell_cell_map = logical_or(cell_cell_maps.values()).astype(uint8)

    visualization = zeros((4, mask.shape[0], mask.shape[1]), dtype=float)
    cell_data_dict = {}
    for cell in unique(mask):
        if cell == BACKGROUND:
            continue
        cell_mask = mask.copy() == cell
        cell_data_dict[int(cell)] = [
            cell_mask.sum(),
            (cell_mask & (cell_cell_map | cell_tissue_map)).sum(),
            (cell_mask & cell_cell_map).sum(),
            (cell_mask & cell_tissue_map).sum(),
        ]
        visualization[0, ...] += (
            getdata(cell_mask) * cell_data_dict[int(cell)][0]
        )
        visualization[1, ...] += (
            getdata(cell_mask & (cell_cell_map | cell_tissue_map)) * cell_data_dict[int(cell)][1]
        )
        visualization[2, ...] += (
            getdata(cell_mask & cell_cell_map) * cell_data_dict[int(cell)][2]
        )
        visualization[3, ...] += (
            getdata(cell_mask & cell_tissue_map) * cell_data_dict[int(cell)][3]
        )
    visualization[0, ...] *= 255 / visualization[0, ...].max()
    visualization[1, ...] *= 255 / visualization[1, ...].max()
    visualization[2, ...] *= 255 / visualization[2, ...].max()
    visualization[3, ...] *= 255 / visualization[3, ...].max()
    #expansions = zeros((mask.shape[0], mask.shape[1]), dtype=uint16)
    #expansions += visualization[0, ...].astype(uint16)
    #expansions += (visualization[1, ...].astype(uint16) << 8).astype(uint16)
    #for _ in range(_width):
    #    expansions = expand(expansions)
    #visualization[0, ...] = ((expansions << 8) >> 8).astype(uint8)
    #visualization[1, ...] = (expansions >> 8).astype(uint8)
    #data = visualization.sum(axis=0) > 0
    #data = data.astype(uint8)
    #data *= 255
    #mask = visualization[[0, 1], ...].sum(axis=0) > 0
    #visualization[2, mask] = 0
    #visualization[4, ...] = data
    visualization = visualization.astype(uint8)
    for idx in range(3):
        _data = rescale(visualization[idx, ...])
        visualization[idx, ...] = _data
    visualization = visualization.transpose(1, 2, 0)
    visualization = visualization.astype(uint8)

    # Generate Area Map
    maps = zeros((mask.shape[0], mask.shape[1], 4))
    for cell in unique(mask):
        if cell == BACKGROUND:
            continue
        for idx in range(4):
            maps[mask == cell, idx] = cell_data_dict[cell][idx]

    return cell_data_dict, maps, visualization


def full_energetics(cell_model_path, nuclear_model_path, image, _width=2, fixup=False):
    """# Compute Energetic Maps

    cell_model_path: Cell-Mask Cellpose Model
    nuclear_model_path: Nuclear-Mask Cellpose Model
    image: Image to get energetics of
    fixup: Whether or not to automatically closeup holes
    """
    from cellpose import models
    from .config import ModelBuildConfig

    cm_cfg = ModelBuildConfig()
    cm_cfg.pretrained_model = cell_model_path
    nm_cfg = ModelBuildConfig()
    nm_cfg.pretrained_model = nuclear_model_path
    cm = models.CellposeModel(**cm_cfg.todict())
    nm = models.CellposeModel(**nm_cfg.todict())
    if fixup:
        from .postprocess import postforward

        c_ma = postforward(cm, image)
    else:
        if isinstance(image, list):
            c_ma = cm.eval(image)[0]
        else:
            c_ma = cm.eval([image])[0]
    if isinstance(image, list):
        n_ma = nm.eval(image)[0]
    else:
        n_ma = nm.eval([image])[0]
    if len(c_ma) == 1:
        c_ma = c_ma[0]
    if len(n_ma) == 1:
        n_ma = n_ma[0]
    c_cdd, c_maps, c_v = compute_energetics(c_ma, _width=_width)
    n_cdd, n_maps, n_v = compute_energetics(n_ma, _width=_width)
    return (c_cdd, n_cdd), (c_maps, n_maps), (c_ma, n_ma), (c_v, n_v)
