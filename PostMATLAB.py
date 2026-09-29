import pandas as pd
from json import loads
from numpy import load, unique, argwhere
from tqdm.auto import tqdm
from skimage.measure import regionprops_table
from os.path import exists

BASE = "tmp_out/MATLAB"
RADIUS = 10
NUCLEI_RPROPERTIES = [
    "label",
    "centroid-0",
    "centroid-1",
    "area",
    "area_convex",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "extent",
    "solidity",
    "equivalent_diameter_area",
]
NUCLEI_PROPERTIES = [
    "label",
    "centroid",
    "area",
    "area_convex",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "extent",
    "solidity",
    "equivalent_diameter_area",
]
meta_map = dict()
with open(f"{BASE}_meta.json", "r") as fhandle:
    meta_map = loads(fhandle.read())

data = pd.read_csv(f"{BASE}.csv")
data["MaskId"] = -1
df = data.copy()

if exists(f"{BASE}.updated.csv"):
    data = pd.read_csv(f"{BASE}.updated.csv")
else:
    for frame_index, name in tqdm(meta_map.items()):
        frame_data = load(name[:-len("mat")] + "npz")["arr_0"]
        unv = unique(frame_data)
        frame = int(frame_index) + 1
        sdf = df[df["Frames"] == frame]
        if len(unv) > sdf.shape[0] + 1:
            pass
            #print(name, frame, sdf.shape, len(unv))
            #print("Count Divergence")
    
        for value in unv:
            if value == 0:
                continue
            centroid = argwhere(frame_data == value).mean(axis=0)
            sdf["D"] = (sdf["CentroidX"] - centroid[1]) ** 2. + (sdf["CentroidY"] - centroid[0]) ** 2.
            min_distance_point = sdf.iloc[sdf["D"].argmin()]
            if min_distance_point["D"] > RADIUS:
                pass
                #print(value, centroid, [sdf[sdf["CellNum"] == value]["CentroidY"].item(), sdf[sdf["CellNum"] == value]["CentroidX"].item()])
                #print(min_distance_point)
            else:
                data_prior = data.loc[(data["Frames"] == frame) & (data["CellNum"] == min_distance_point["CellNum"]),"MaskId"].item()
                data.loc[(data["Frames"] == frame) & (data["CellNum"] == min_distance_point["CellNum"]),"MaskId"] = value
                data_post = data.loc[(data["Frames"] == frame) & (data["CellNum"] == min_distance_point["CellNum"]),"MaskId"].item()

    df = data.copy()
    df["CellNum"] = data["MaskId"]
    df["MaskId"] = data["CellNum"]
    df.to_csv(f"{BASE}.updated.csv")
    data = df.copy()
df = data.copy()

df["Energy-IP"] = 0.
df["Energy-ST"] = 0.
df["Energy-CCA"] = 0.
df["Energy-CSA"] = 0.
for prop in NUCLEI_RPROPERTIES:
    df[f"Nuclei-{prop}"] = 0.

for frame_index, name in tqdm(meta_map.items()):
    if exists(name[: -len(".mat")] + ".Full.csv"):
        continue
    frame = int(frame_index) + 1
    sdf = df[df["Frames"] == frame]
    frame_data = load(name[: -len("mat")] + "npz")["arr_0"]
    # Load Nuclei Data
    nuclei_frame_data = load(name[: -len("MembraneSeg.mat")] + "NucleiSeg.npz")["arr_0"]
    # Load Energy Data
    cell_cell_frame_data = load(
        name[: -len("MembraneSeg.mat")] + "MembraneEnergy0.npz"
    )["arr_0"]
    cell_stroma_frame_data = load(
        name[: -len("MembraneSeg.mat")] + "MembraneEnergy2.npz"
    )["arr_0"]
    perimeter_frame_data = load(
        name[: -len("MembraneSeg.mat")] + "MembraneEnergy1.npz"
    )["arr_0"]

    cells = unique(sdf["CellNum"])
    nrp = regionprops_table(nuclei_frame_data, properties=NUCLEI_PROPERTIES)
    nrp_df = pd.DataFrame(nrp)
    for cell in cells:
        if cell == 0:
            assert False, "Should not reach here"
        cell_mask = frame_data == cell
        nfd = nuclei_frame_data.copy()
        nfd[~cell_mask] = 0
        ccfd = cell_cell_frame_data & cell_mask
        csfd = cell_stroma_frame_data & cell_mask
        pfd = perimeter_frame_data & cell_mask
        E_A = cell_mask.sum()
        E_P = pfd.sum()
        E_CC = ccfd.sum()
        E_CS = csfd.sum()
        df.loc[(df["Frames"] == frame) & (df["CellNum"] == cell), "Energy-IP"] = float(E_A)
        df.loc[(df["Frames"] == frame) & (df["CellNum"] == cell), "Energy-ST"] = float(E_P)
        df.loc[(df["Frames"] == frame) & (df["CellNum"] == cell), "Energy-CCA"] = float(E_CC)
        df.loc[(df["Frames"] == frame) & (df["CellNum"] == cell), "Energy-CSA"] = float(E_CS)
        nv, nc = unique(nfd, return_counts=True)
        ndict = dict(zip(nv, nc))
        del ndict[0]
        if len(ndict.keys()) == 0:
            continue
        max_nuclei = max(ndict, key=ndict.get)  # pyright: ignore
        nuclei_mask = nuclei_frame_data == max_nuclei
        nrp_series = nrp_df[nrp_df["label"] == max_nuclei]
        assert nrp_series.shape[0] == 1
        nrp_series = nrp_series.iloc[0]
        for _property in NUCLEI_RPROPERTIES:
            df.loc[(df["Frames"] == frame) & (df["CellNum"] == cell), f"Nuclei-{_property}"] = nrp_series[_property]
    sdf = df[df["Frames"] == frame]
    sdf.to_csv(name[: -len(".mat")] + ".Full.csv", index=False)

#for frame_index, name in tqdm(meta_map.items()):
#    frame = int(frame_index) + 1
#    sdf = df[df["Frames"] == frame]
#    sdf.to_csv(name[: -len("mat")] + "csv", index=False)

#print(data)
