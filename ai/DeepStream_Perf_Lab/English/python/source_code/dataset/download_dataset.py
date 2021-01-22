import gdown
import os
## Video WildTrack DATASET
url = 'https://drive.google.com/u/0/uc?id=1lE8oky9hZ0v552Il29WoULyE93sxjN_v&export=download'
output = '/opt/nvidia/deepstream/deepstream-5.0/python/source_code/dataset/deepstream_dataset.zip'
gdown.download(url, output, quiet=False,proxy=None)
