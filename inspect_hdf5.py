import h5py
import sys

path = sys.argv[1]

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"GROUP:   {name}")

with h5py.File(path, "r") as f:
    f.visititems(walk)