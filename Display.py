from matplotlib import pyplot as plt
from numpy import array, unique, load, vectorize
from numpy.ma import masked_array
from utils import *
from histseg.tools.scanner2d import Scanner2D as Scanner
from histseg.tools.tree_workers import *
from time import perf_counter

NAME="HT-B5"
NAME="HT-C5"
NAME="HT-C7"
OUTPUT_BASE = f"tmp_out/Display/{NAME}"
image = plt.imread(f"../../home/Downloads/Images/{NAME.replace('HT-','')}.jpg")
c_ma, n_ma, _, _ = segment(image, "promising/Model-C00.pt", "promising/Model-N00.pt")
plt.imsave(f"{OUTPUT_BASE}-C.png", c_ma)
plt.imsave(f"{OUTPUT_BASE}-N.png", n_ma)

# Get Cell Outlines
scanner_cell = Scanner(c_ma)
worker_cell = TreeWorker(scanner_cell, width=3)
worker_cell.new_worker(InteriorOutlineSubworker)
worker_cell.new_worker(FullOutlineSubworker)
worker_cell.new_worker(ExteriorOutlineSubworker)
scanner_cell.assign(worker_cell)
scanner_cell.scan()
for index, data in enumerate(worker_cell.new_data):
    plt.imsave(f"{OUTPUT_BASE}-C{index}.png", np.ma.masked_equal(data, 0), cmap='Reds', vmin=0, vmax=2)

# Get Cell Outlines
scanner_nucleus = Scanner(n_ma)
worker_nucleus = TreeWorker(scanner_nucleus, width=3)
worker_nucleus.new_worker(InteriorOutlineSubworker)
worker_nucleus.new_worker(FullOutlineSubworker)
worker_nucleus.new_worker(ExteriorOutlineSubworker)
scanner_nucleus.assign(worker_nucleus)
scanner_nucleus.scan()

for index, data in enumerate(worker_nucleus.new_data):
    plt.imsave(f"{OUTPUT_BASE}-N{index}.png", np.ma.masked_equal(data, 0), cmap='Blues', vmin=0, vmax=2)

data = load(
    f"../../Data/ihc/data/segmentation/histology-260127/train/{NAME}_seg.npy", allow_pickle=True
).item()
source = data["masks"]

scanner = Scanner(source)
worker = TreeWorker(scanner, width=3)
worker.new_worker(InteriorOutlineSubworker)
worker.new_worker(FullOutlineSubworker)
worker.new_worker(ExteriorOutlineSubworker)
scanner.assign(worker)

counter = 0
t0 = perf_counter()
for y in range(source.shape[1]):
    for x in range(source.shape[0]):
        counter += source[x, y]
t1 = perf_counter()
print(t1 - t0, counter)
t0 = perf_counter()
scanner.scan(progress=True)
t1 = perf_counter()
print(t1 - t0)
# print(source)
# for data in worker.new_data:
#    print(data)
for index, data in enumerate(worker.new_data):
    plt.imsave(f"{OUTPUT_BASE}{index}.png", data)

_masks = source
_pres_n = masked_array(_masks, (_masks == 0).astype(bool)).filled(-1)
_surf_n = masked_array(_masks, ~worker.new_data[1].astype(bool)).filled(-1)
_ccsu_n = masked_array(_masks, ~worker.new_data[0].astype(bool)).filled(-1)
_cssu_n = masked_array(_masks, ~worker.new_data[2].astype(bool)).filled(-1)

# Getting energetics from results
Pressure = array(unique(_pres_n, return_counts=True)).T[:-1, :]
Tension = array(unique(_surf_n, return_counts=True)).T[:-1, :]
CancerAdhesion = array(unique(_ccsu_n, return_counts=True)).T[:-1, :]
StromaAdhesion = array(unique(_cssu_n, return_counts=True)).T[:-1, :]
print(Pressure[:10])
print(Tension[:10])
print(CancerAdhesion[:10])
print(StromaAdhesion[:10])

_Pressure = {k: v for (k, v) in Pressure}
for k in unique(source):
    if k not in _Pressure.keys():
        _Pressure[k] = 0
_Tension = {k: v for (k, v) in Tension}
for k in unique(source):
    if k not in _Tension.keys():
        _Tension[k] = 0
_CancerAdhesion = {k: v for (k, v) in CancerAdhesion}
for k in unique(source):
    if k not in _CancerAdhesion.keys():
        _CancerAdhesion[k] = 0
_StromaAdhesion = {k: v for (k, v) in StromaAdhesion}
for k in unique(source):
    if k not in _StromaAdhesion.keys():
        _StromaAdhesion[k] = 0
_Pressure = vectorize(_Pressure.get)
_Tension = vectorize(_Tension.get)
_CancerAdhesion = vectorize(_CancerAdhesion.get)
_StromaAdhesion = vectorize(_StromaAdhesion.get)

for index, (mask, fmap) in enumerate(
    [(source == 0, _Pressure), (~worker.new_data[1].astype(bool), _Tension), (~worker.new_data[0].astype(bool), _CancerAdhesion), (~worker.new_data[2].astype(bool), _StromaAdhesion)]
):
    # _data = fmap(masked_array(source, source==0))
    _data = fmap(masked_array(source, mask))
    #print(_data)
    plt.imsave(f"{OUTPUT_BASE}-E{index}.png", _data)
