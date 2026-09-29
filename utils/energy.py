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


def compute_energetics(mask):
    """# Compute the Energetics of a Mask

    mask: (W,H) of integer type

    returns: (W,H,4)
    Where the 4 channels are the pressure, tension, inter-mask adhesion, mask-background adhesion
    """
    if isinstance(mask, list):
        return [compute_energetics(m) for m in mask]
    from scipy.signal import convolve2d
    from numpy import uint8, unique, zeros

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

    # Generate Area Map
    maps = zeros((mask.shape[0], mask.shape[1], 4))
    for cell in unique(mask):
        if cell == BACKGROUND:
            continue
        for idx in range(4):
            maps[mask == cell, idx] = cell_data_dict[cell][idx]

    return cell_data_dict, maps


def full_energetics(cell_model_path, nuclear_model_path, image, fixup=False):
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
    c_cdd, c_maps = compute_energetics(c_ma)
    n_cdd, n_maps = compute_energetics(n_ma)
    return (c_cdd, n_cdd), (c_maps, n_maps)
