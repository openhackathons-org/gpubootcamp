# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

import gdown
import os

## VOC dataset
# uncomment the file
url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

# if the url above fails, uncomment the url below and rebiuld your container
#url = 'https://drive.google.com/file/d/1YkakD_qJuKmwAZF_HwLGLNb8fcqZbtx7/view?usp=sharing'
output_python = '/workspace/tlt-experiments/data/VOCtrainval_11-May-2012.tar'
gdown.download(url, output_python, quiet=False, proxy=None)
