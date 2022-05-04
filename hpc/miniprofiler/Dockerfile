# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build: $ sudo docker build -t nvidia_nsight_profiling_openacc:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 nvidia_nsight_profiling_openacc:latest
# Finally, open http://127.0.0.1:8888/

FROM nvcr.io/nvidia/nvhpc:22.3-devel-cuda_multi-ubuntu20.04

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install  --no-install-recommends \
    m4 vim-nox emacs-nox nano zip\
    python3-pip python3-setuptools git-core inotify-tools \
    curl git-lfs nginx\
    build-essential openssh-server openssh-client && \
    rm -rf /var/lib/apt/cache/* 

RUN apt-get update 
RUN apt-get install --no-install-recommends -y python3
RUN pip3 install --upgrade pip
RUN apt-get update -y        
RUN apt-get install -y git nvidia-modprobe
RUN pip3 install jupyterlab
# Install required python packages
RUN pip3 install ipywidgets
RUN pip3 install netcdf4

RUN apt-get install --no-install-recommends -y build-essential

ENV PATH="$PATH:/usr/local/bin:/opt/anaconda3/bin:/usr/bin" 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3  && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/anaconda3/bin/conda install -y -q netcdf4


ADD English/ /labs
WORKDIR /labs
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs