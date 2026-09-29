from matplotlib import pyplot as plt
from numpy import array, unique, load, vectorize
from numpy.ma import masked_array
from histseg.tools.scanner2d import Scanner2D as Scanner
from histseg.tools.tree_workers import *
from time import perf_counter

# source = array([
#        [0,0,0,0,0,0,0,0,0,0,0,],
#        [0,1,1,1,2,2,2,2,0,0,0,],
#        [1,1,1,1,2,2,2,2,0,0,0,],
#        [1,1,1,1,1,1,2,2,0,0,0,],
#        [3,3,3,3,3,1,2,2,0,0,0,],
#        [3,3,3,3,3,1,2,2,0,0,0,],
#        [3,3,3,3,3,2,2,2,0,0,0,],
#        [3,3,3,3,3,3,3,3,3,3,3,],
#        [3,3,3,3,3,3,3,3,3,3,3,],
#        [3,3,3,3,3,3,3,3,3,3,3,],
#        [3,3,3,3,3,3,3,3,3,3,3,],
#        ])
data = load(
    "../../Data/ihc/data/segmentation/train/HT-B5_seg.npy", allow_pickle=True
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
    plt.imsave(f"../{index}.png", data)

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
    plt.imsave(f"../E{index}.png", _data)
