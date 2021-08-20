# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

import gdown
import os

##  input files in data.zip
url = 'https://drive.google.com/uc?id=1WZ0rtXZ-uMLfy7htT0gaU4EQ_Rq61QTF&export=download' ##remote URL for dataset location, need internet access to download
output_python = '/bootcamp/datasets/data.zip'
gdown.download(url, output_python, quiet=False, proxy=None)
os.system("unzip data.zip -d /bootcamp/datasets/")          ##path to dataset folder, modify as required
