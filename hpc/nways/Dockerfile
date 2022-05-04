# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $ sudo docker build -t nways-labs:latest .
# To run: $ sudo docker run --rm -it --runtime nvidia -p 8888:8888 nways-labs:latest
# Finally, open http://localhost:8888/

FROM nvcr.io/nvidia/nvhpc:22.3-devel-cuda_multi-ubuntu20.04

RUN apt-get -y update && \
        DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip make build-essential libtbb-dev && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --upgrade pip &&\
        pip3 install gdown
        
RUN apt-get update -y        
RUN apt-get install -y git nvidia-modprobe
RUN pip3 install jupyterlab
# Install required python packages
RUN pip3 install ipywidgets

############################################
RUN apt-get update -y

# TO COPY the data
COPY nways_labs/ /labs/

RUN python3 /labs/nways_MD/English/C/source_code/dataset.py
RUN python3 /labs/nways_MD/English/Fortran/source_code/dataset.py

#################################################
ENV PATH="/usr/local/bin:/opt/anaconda3/bin:/usr/bin:$PATH"
#################################################

ADD nways_labs/ /labs
WORKDIR /labs
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs
