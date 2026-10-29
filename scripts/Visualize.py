#!/usr/bin/env python
from matplotlib import pyplot as plt
from histseg.utils import *

IMAGE_BASE_DIRECTORY = "../../home/Downloads/Work/Images/"
OUTPUT_BASE_DIRECTORY = "FigurePieces/"

CMAP_C = "summer"
CMAP_N = "cool"


def _process(im_path):
    im = load_image(im_path)

    im_name = im_path.split("/")[-1]
    output_bdir = im_name
    # Strip any extension
    if len(im_name.split(".")) > 1:
        ext = im_name.split("/")[-1].split(".")[-1]
        output_bdir = im_name[: -len(f".{ext}")]
    output_bdir = f"{OUTPUT_BASE_DIRECTORY}/{output_bdir}"

    c_ma_unproc, n_ma_unproc, _, _ = segment(
        im,
        "promising/Model-C00.pt",
        "promising/Model-N00.pt",
        restrict=False,
        closeup=False,
        masked=True,
    )
    c_ma_proc, n_ma_proc, _, _ = segment(
        im, "promising/Model-C00.pt", "promising/Model-N00.pt", masked=True
    )

    # Unprocessed
    plt.imsave(
        f"{output_bdir}_CellSeg.Unprocessed.png", color_mask(c_ma_unproc), cmap=CMAP_C
    )
    plt.imsave(
        f"{output_bdir}_NucleiSeg.Unprocessed.png",
        mask_ncolor(n_ma_unproc),
        cmap=CMAP_N,
    )
    # Postprocessed
    plt.imsave(
        f"{output_bdir}_CellSeg.Processed.png", color_mask(c_ma_proc), cmap=CMAP_C
    )
    plt.imsave(
        f"{output_bdir}_Nuclei.Processed.png", mask_ncolor(n_ma_proc), cmap=CMAP_N
    )

    # Energetics Visualization
    _, _, _, c_v = full_energetics(
        "promising/Model-C00.pt", "promising/Model-N00.pt", im, _width=2, fixup=True
    )
    plt.imsave(f"{output_bdir}_Energy.png", c_v[0])

    # Using thinner lines
    _, _, _, c_v = full_energetics(
        "promising/Model-C00.pt", "promising/Model-N00.pt", im, _width=1, fixup=True
    )
    plt.imsave(f"{output_bdir}_Energy.Thin.png", c_v[0])


for name in ["B5", "B9", "C7"]:
    im_path = f"{IMAGE_BASE_DIRECTORY}{name}.jpg"
    print(f"Visualizing {im_path}...")
    _process(im_path)
    print(f"Visualized {im_path}!")
