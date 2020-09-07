# Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gdown
import os
## CFD TRAIN DATASET
url = 'https://drive.google.com/uc?id=0BzsbU65NgrSuZDBMOW93OWpsMHM&export=download'
output = '/workspace/python/jupyter_notebook/CFD/data/train.tfrecords'
gdown.download(url, output, quiet=False,proxy=None)

## CFD TEST DATASET
url = 'https://drive.google.com/uc?id=1WSJLK0cOQehixJ6Tf5k0eYDcb4RJ5mXv&export=download'
output = '/workspace/python/jupyter_notebook/CFD/data/test.tfrecords'
gdown.download(url, output, quiet=False,proxy=None)

## CFD CONV_SDF MODEL
url = 'https://drive.google.com/uc?id=1pfR0io1CZKvXArGk-nt2wciUoAN_6Z08&export=download'
output = '/workspace/python/jupyter_notebook/CFD/conv_sdf_model.h5'
gdown.download(url, output, quiet=False,proxy=None)

## CFD CONV MODEL
url = 'https://drive.google.com/uc?id=1rFhqlQnTkzIyZocjAxMffucmS3FDI0_j&export=download'
output = '/workspace/python/jupyter_notebook/CFD/conv_model.h5'
gdown.download(url, output, quiet=False,proxy=None)


## CFD TEST Dataset
url = 'https://drive.google.com/uc?id=0BzsbU65NgrSuR2NRRjBRMDVHaDQ&export=download'
output = '/workspace/python/jupyter_notebook/CFD/data/computed_car_flow.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
