# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

import gdown
import os

## alk.traj.dcd input file 
url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
output_python = '/workspace/tlt-experiments/data/VOCtrainval_11-May-2012.tar'
gdown.download(url, output_python, quiet=False, proxy=None)
