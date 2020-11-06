# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build: $ sudo docker build -t nvidia_nsight_profiling_openacc:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 nvidia_nsight_profiling_openacc:latest
# Finally, open http://127.0.0.1:8888/

FROM nvcr.io/nvidia/nvhpc:20.9-devel-ubuntu20.04

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

RUN pip3 install --no-cache-dir jupyter
RUN pip3 install netcdf4

# NVIDIA nsight-systems-2020.2.1 
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    wget && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F60F4B3D7FA2AF80 && \
    echo "deb https://developer.download.nvidia.com/devtools/repo-deb/x86_64/ /" >> /etc/apt/sources.list.d/nsight.list && \
    apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2020.2.1

RUN apt-get install --no-install-recommends -y build-essential

ENV PATH="$PATH:/usr/local/bin:/opt/anaconda3/bin:/usr/bin" 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3  && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/anaconda3/bin/conda install -y -q netcdf4

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/lib64/"
ENV PATH="/opt/nvidia/nsight-systems/2020.2.1/bin:$PATH"

ADD English/ /labs
WORKDIR /labs
CMD service nginx start && jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs