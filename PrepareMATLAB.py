from scipy.io import loadmat, savemat
from utils import *
from histseg.tools.tree_workers import *
from os import link
from json import dumps
#from shutil import copyfile

from glob import glob

BASE = "tmp_out/MATLAB"
files = glob("../../Data/ihc/data/classification/*.*/HistSeg-MembraneSeg.mat")
files = sorted(files)

meta_map = dict()
for idx,ffile in enumerate(files):
    link(ffile, f"{BASE}/{idx:03}.mat")
    meta_map[idx] = ffile

with open(f"{BASE}_meta.json", "w+") as fhandle:
    fhandle.write(dumps(meta_map, indent=2))

print(f"Prepared to: {BASE}")
