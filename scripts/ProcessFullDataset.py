from scipy.io import savemat
from matplotlib import pyplot as plt
from histseg.utils import *
from histseg.tools.scanner2d import Scanner2D as Scanner
from histseg.tools.tree_workers import *

from tqdm.auto import tqdm
from glob import glob

base_dirs = glob("../../Data/ihc/data/classification/*.*/")

# Get the membrane and nuclei segmentations
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

# Get the energetics
