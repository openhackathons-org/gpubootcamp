# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

import gdown
import os

## VOC dataset

url0 = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

# if the url0 above fails, uncomment the url0 below and rebiuld your container
#url0 = 'https://drive.google.com/file/d/1YkakD_qJuKmwAZF_HwLGLNb8fcqZbtx7/view?usp=sharing'

output_python0 = '/workspace/tlt-experiments/data/VOCtrainval_11-May-2012.tar'

## Kitti detection dataset for yolov4
url1 = 'http://www.cvlibs.net/download.php?file=data_object_label_2.zip'
url2 = 'http://www.cvlibs.net/download.php?file=data_object_image_2.zip'

output_python1 = '/workspace/tlt-experiments/data/data_object_label_2.zip'
output_python2 = '/workspace/tlt-experiments/data/data_object_image_2.zip'

gdown.download(url0, output_python0, quiet=False, proxy=None)
gdown.download(url1, output_python1, quiet=False, proxy=None)
gdown.download(url2, output_python2, quiet=False, proxy=None)
