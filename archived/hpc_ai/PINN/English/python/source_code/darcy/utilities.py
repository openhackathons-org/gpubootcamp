import os
import zipfile

try:
    import gdown
except:
    gdown = None

import scipy.io
import numpy as np
import h5py

from modulus.hydra import to_absolute_path

# list of FNO dataset url ids on drive: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
_FNO_datatsets_ids = {
    "Darcy_241": "1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV",
    "Darcy_421": "1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf",
}
_FNO_dataset_names = {
    "Darcy_241": (
        "piececonst_r241_N1024_smooth1.hdf5",
        "piececonst_r241_N1024_smooth2.hdf5",
    ),
    "Darcy_421": (
        "piececonst_r421_N1024_smooth1.hdf5",
        "piececonst_r421_N1024_smooth2.hdf5",
    ),
}


def load_FNO_dataset(path, input_keys, output_keys, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar)


def download_FNO_dataset(name, outdir="datasets/"):
    "Tries to download FNO dataset from drive"

    if name not in _FNO_datatsets_ids:
        raise Exception(
            f"Error: FNO dataset {name} not recognised, select one from {list(_FNO_datatsets_ids.keys())}"
        )

    id = _FNO_datatsets_ids[name]
    outdir = to_absolute_path(outdir) + "/"
    namedir = f"{outdir}{name}/"

    # skip if already exists
    exists = True
    for file_name in _FNO_dataset_names[name]:
        if not os.path.isfile(namedir + file_name):
            exists = False
            break
    if exists:
        return
    print(f"FNO dataset {name} not detected, downloading dataset")

    # Make sure we have gdown installed
    if gdown is None:
        raise ModuleNotFoundError("gdown package is required to download the dataset!")

    # get output directory
    os.makedirs(namedir, exist_ok=True)

    # download dataset
    zippath = f"{outdir}{name}.zip"
    _download_file_from_google_drive(id, zippath)

    # unzip
    with zipfile.ZipFile(zippath, "r") as f:
        f.extractall(namedir)
    os.remove(zippath)

    # preprocess files
    for file in os.listdir(namedir):
        if file.endswith(".mat"):
            matpath = f"{namedir}{file}"
            preprocess_FNO_mat(matpath)
            os.remove(matpath)


def _download_file_from_google_drive(id, path):
    "Downloads a file from google drive"

    # use gdown library to download file
    gdown.download(id=id, output=path)


def preprocess_FNO_mat(path):
    "Convert a FNO .mat file to a hdf5 file, adding extra dimension to data arrays"

    assert path.endswith(".mat")
    data = scipy.io.loadmat(path)
    ks = [k for k in data.keys() if not k.startswith("__")]
    with h5py.File(path[:-4] + ".hdf5", "w") as f:
        for k in ks:
            x = np.expand_dims(data[k], axis=1)  # N, C, H, W
            f.create_dataset(
                k, data=x, dtype="float32"
            )  # note h5 files larger than .mat because no compression used
