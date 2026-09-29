import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from ciecam02 import rgb2jch, jch2rgb
from json import dumps


def generate_synthetic_stain(
    width=512,
    height=512,
    tess_avg_area=2000,
    effect_max=(0.45, 0.45, 0.35),
    noise_level=0.15,
    palette_stride=1,
    gen_colors=512,
    nuclei_dist_max=12,
    n_background=80,
    save_img=False,
):
    # * Step 0
    WIDTH, HEIGHT = width, height
    # TESSELATION_AVERAGE_AREA = 350
    TESSELATION_AVERAGE_AREA = tess_avg_area
    EFFECT_MAX = effect_max
    NOISE_LEVEL = noise_level
    PALETTE_STRIDE = palette_stride
    GEN_COLORS = gen_colors
    # GEN_COLORS = 8
    TOTAL_AREA = WIDTH * HEIGHT
    TOTAL_CELL_COUNT = int(np.floor(TOTAL_AREA / TESSELATION_AVERAGE_AREA))
    NUCLEI_DISTANCE_MAX = nuclei_dist_max
    N_BACKGROUND = n_background
    JCH_SORT = 0, 2, 1

    # * Step 1
    primary_palettes = {
        "cell_primary": (
            np.array(
                [
                    (145, 110, 57),
                    (187, 166, 125),
                    (172, 150, 100),
                ],
                dtype=np.double,
            )
            / 255
        ).copy(),
        "nuclei_primary": (
            np.array(
                [
                    (106, 87, 118),
                    (45, 32, 84),
                    #                (117, 119, 129),
                    (0, 0, 128),
                ],
                dtype=np.double,
            )
            / 255
        ).copy(),
        "background_primary": (
            np.array(
                [
                    (214, 221, 224),
                    (255, 255, 255),
                    (230, 231, 238),
                ],
                dtype=np.double,
            )
            / 255
        ).copy(),
    }
    # print(dumps({p:(v*255).astype(int).tolist() for p,v in primary_palettes.items()}, indent=2))

    full_palettes = {p: None for p in primary_palettes.keys()}
    for pidx, primary_s in enumerate(primary_palettes.keys()):
        palettes = []
        for primary_idx, primary in enumerate(primary_palettes[primary_s]):
            remaining_indicies = list(
                set(range(len(primary_palettes[primary_s])))
                - {
                    primary_idx,
                }
            )
            primary_color = primary.copy()
            palette = [
                primary_color.copy().tolist(),
            ]
            for _ in range(GEN_COLORS - 1):
                d_color = primary_palettes[primary_s].copy()[
                    np.random.choice(remaining_indicies).item()
                ]
                effect_size = (np.random.rand() * 2 - 1) * EFFECT_MAX[pidx]
                d_color *= effect_size
                color = primary_color.copy() + d_color
                palette.append(color.tolist())
            palette = np.array(palette)
            palette = np.clip((palette * 255).astype(int), 0, 255)
            jch_palette = rgb2jch(palette)
            jch_idxs = np.lexsort(
                (
                    jch_palette[:, JCH_SORT[0]],
                    jch_palette[:, JCH_SORT[1]],
                    jch_palette[:, JCH_SORT[2]],
                )
            )
            jch_sorted = jch_palette[jch_idxs]
            rgb_palette = jch2rgb(jch_sorted)[::PALETTE_STRIDE, ...]
            palettes.append(np.unique(rgb_palette.copy(), axis=0))
        full_palettes[primary_s] = palettes.copy()
    # print([[vv.shape for vv in v] for v in full_palettes.values()])

    ## CELL
    # * Step 2
    cell_centers = np.random.randint(
        (0, 0), (WIDTH, HEIGHT), size=(TOTAL_CELL_COUNT, 2)
    )
    distance_tree = KDTree(cell_centers)

    coordinates = np.mgrid[:WIDTH, :HEIGHT].reshape((2, -1)).T
    distance_map, cell_map = distance_tree.query(coordinates, 1)
    distance_map = distance_map.reshape((WIDTH, HEIGHT)).astype(np.int64)
    cell_map = cell_map.reshape((WIDTH, HEIGHT))
    if save_img:
        plt.imsave("distance.png", distance_map)
        plt.imsave("cell.png", cell_map)

    # * Step 3
    palette = primary_palettes["cell_primary"]
    cells = np.unique(cell_map)
    cell_remap_values = np.random.randint(palette.shape[0], size=cells.shape[0])
    cell_remap = {
        cells[idx].item(): cell_remap_values[idx].item()
        for idx in range(cells.shape[0])
    }

    @np.vectorize
    def _apply(x):
        return cell_remap[x]

    primary_palette_map = _apply(cell_map)
    if save_img:
        plt.imsave("primary.png", primary_palette_map)

    # * Step 4
    MAX_DISTANCE = np.max(distance_map)
    palette = full_palettes["cell_primary"]
    MIN_PALETTE_SIZE = min([p.shape[0] for p in palette])
    MAX_START = MIN_PALETTE_SIZE - MAX_DISTANCE - 1
    UNIQUE_CELLS = np.unique(cells)
    CELL_DISTANCE_START = np.random.randint(MAX_START, size=UNIQUE_CELLS.shape[0])

    grad_colors = np.zeros((WIDTH, HEIGHT, 3))

    for x in range(WIDTH):
        for y in range(HEIGHT):
            _cell = np.where(UNIQUE_CELLS == cell_map[x, y])[0]
            _cell = _cell.item()
            grad_colors[x, y, :] = palette[primary_palette_map[x, y]][
                CELL_DISTANCE_START[_cell] + distance_map[x, y], :
            ]
    if save_img:
        plt.imsave("grad.png", grad_colors / 255)

    ## * Step 4
    # noise = grad_colors.copy()/255 + np.random.rand(*grad_colors.shape) * NOISE_LEVEL
    # plt.imsave("noise.png", np.clip(noise, 0, 1))

    ## Nuclei
    # * Step 3
    palette = primary_palettes["nuclei_primary"]
    cells = np.unique(cell_map)
    nucl_remap_values = np.random.randint(palette.shape[0], size=cells.shape[0])
    nucl_remap = {
        cells[idx].item(): nucl_remap_values[idx].item()
        for idx in range(cells.shape[0])
    }

    @np.vectorize
    def _apply(x):
        return nucl_remap[x]

    nucl_map = _apply(cell_map)
    if save_img:
        plt.imsave("nucl.png", nucl_map)

    # * Step 4
    palette = full_palettes["nuclei_primary"]
    MIN_PALETTE_SIZE = min([p.shape[0] for p in palette])
    MAX_START = MIN_PALETTE_SIZE - MAX_DISTANCE - 1
    NUCL_DISTANCE_START = np.random.randint(MAX_START, size=UNIQUE_CELLS.shape[0])

    # grad_colors = np.zeros((WIDTH, HEIGHT, 3))

    for x in range(WIDTH):
        for y in range(HEIGHT):
            if distance_map[x, y] < NUCLEI_DISTANCE_MAX:
                _cell = np.where(UNIQUE_CELLS == cell_map[x, y])[0]
                _cell = _cell.item()
                grad_colors[x, y, :] = palette[nucl_map[x, y]][
                    NUCL_DISTANCE_START[_cell] + distance_map[x, y], :
                ]
    if save_img:
        plt.imsave("grad.png", grad_colors / 255)

    ## # * Step 4
    ## noise = grad_colors.copy()/255 + np.random.rand(*grad_colors.shape) * NOISE_LEVEL
    ## plt.imsave("noise.png", np.clip(noise, 0, 1))

    ## Background
    # * Step 3
    palette = primary_palettes["background_primary"]
    cells = np.unique(cell_map)
    back_remap_values = np.random.randint(palette.shape[0], size=cells.shape[0])
    back_remap = {
        cells[idx].item(): back_remap_values[idx].item()
        for idx in range(cells.shape[0])
    }

    @np.vectorize
    def _apply(x):
        return back_remap[x]

    back_map = _apply(cell_map)
    if save_img:
        plt.imsave("back.png", back_map)

    # * Step 4
    palette = full_palettes["background_primary"]
    MIN_PALETTE_SIZE = min([p.shape[0] for p in palette])
    MAX_START = MIN_PALETTE_SIZE - MAX_DISTANCE - 1
    BACK_DISTANCE_START = np.random.randint(MAX_START, size=UNIQUE_CELLS.shape[0])

    # grad_colors = np.zeros((WIDTH, HEIGHT, 3))

    background_cells = np.random.randint(len(cells), size=(N_BACKGROUND))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if cell_map[x, y] in background_cells:
                _cell = np.where(UNIQUE_CELLS == cell_map[x, y])[0]
                _cell = _cell.item()
                grad_colors[x, y, :] = palette[back_map[x, y]][
                    BACK_DISTANCE_START[_cell] + distance_map[x, y], :
                ]
                cell_map[x, y] = -1
    if save_img:
        plt.imsave("grad.png", grad_colors / 255)

    # * Step 4
    MIN_NOISE = 4
    input_noise = np.random.rand(*distance_map.shape, 3)
    for i in range(3):
        input_noise[..., i] *= (
            (distance_map + MIN_NOISE) / (MAX_DISTANCE + MIN_NOISE)
        ) ** 0.4
    noise = grad_colors.copy() / 255 + input_noise * NOISE_LEVEL
    if save_img:
        plt.imsave("noise.png", np.clip(noise, 0, 1))
    resulting_image = np.clip(noise, 0, 1)
    cell_map += 1
    return resulting_image, cell_map
