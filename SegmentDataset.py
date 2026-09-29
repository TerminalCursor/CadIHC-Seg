from scipy.io import savemat
from matplotlib import pyplot as plt
from utils import *
from histseg.tools.scanner2d import Scanner2D as Scanner
from histseg.tools.tree_workers import *

from tqdm.auto import tqdm
from glob import glob

base_dirs = glob("../../Data/ihc/data/classification/*.*/")

pbar = tqdm(base_dirs)
for base_dir in pbar:
    pbar.set_description(base_dir)
    filename = base_dir + "image.png"
    image = load_image(filename)
    c_ma, n_ma, _, _ = segment(
        image, "promising/Model-C00.pt", "promising/Model-N00.pt"
    )
    v2_naming_base = f"{base_dir}HistSeg-"
    plt.imsave(v2_naming_base + "MembraneSeg.png", color_mask(c_ma))
    np.savez(v2_naming_base + "MembraneSeg.npz", c_ma)
    if image.shape[0] <= 4:
        image = image.transpose(1,2,0)
    savemat(v2_naming_base + "MembraneSeg.mat", dict(labels=c_ma, image=image))
    plt.imsave(v2_naming_base + "NucleiSeg.png", color_mask(n_ma))
    np.savez(v2_naming_base + "NucleiSeg.npz", n_ma)

    # Get Cell Outlines
    scanner_cell = Scanner(c_ma)
    worker_cell = TreeWorker(scanner_cell, width=3)
    worker_cell.new_worker(InteriorOutlineSubworker)
    worker_cell.new_worker(FullOutlineSubworker)
    worker_cell.new_worker(ExteriorOutlineSubworker)
    scanner_cell.assign(worker_cell)
    scanner_cell.scan()
    for index, data in enumerate(worker_cell.new_data):
        np.savez(v2_naming_base + f"MembraneEnergy{index}.npz", data)
        plt.imsave(
            v2_naming_base + f"MembraneEnergy{index}.png",
            np.ma.masked_equal(data, 0),
            cmap="Reds",
            vmin=0,
            vmax=2,
        )

    # Get Nuclear Outlines
    scanner_nucleus = Scanner(n_ma)
    worker_nucleus = TreeWorker(scanner_nucleus, width=3)
    worker_nucleus.new_worker(InteriorOutlineSubworker)
    worker_nucleus.new_worker(FullOutlineSubworker)
    worker_nucleus.new_worker(ExteriorOutlineSubworker)
    scanner_nucleus.assign(worker_nucleus)
    scanner_nucleus.scan()

    for index, data in enumerate(worker_nucleus.new_data):
        np.savez(v2_naming_base + f"NucleiEnergy{index}.npz", data)
        plt.imsave(
            v2_naming_base + f"NucleiEnergy{index}.png",
            np.ma.masked_equal(data, 0),
            cmap="Blues",
            vmin=0,
            vmax=2,
        )
