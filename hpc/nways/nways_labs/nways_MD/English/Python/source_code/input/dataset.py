# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

import gdown
import os

## alk.traj.dcd input file 
url = 'https://drive.google.com/uc?id=1WZ0rtXZ-uMLfy7htT0gaU4EQ_Rq61QTF&export=download'
output = '/labs/nways_MD/English/C/source_code/input/alk.traj.dcd'
gdown.download(url, output, quiet=False,proxy=None)