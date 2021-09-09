# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# Select Base Image
FROM nvcr.io/nvidia/deepstream:5.0.1-20.09-triton

# NVIDIA Nsight Systems 2020.2.1
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/devtools/repo-deb/x86_64/ /" >> /etc/apt/sources.list.d/nsight.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-systems-2020.2.1 && \
    rm -rf /var/lib/apt/lists/* 
# Update the repo
RUN apt-get update
# Install required dependencies
RUN apt-get install ffmpeg python3-gi python3-dev python3-pip cmake unzip -y
# Install pybind11
RUN pip3 install pybind11
# Install required python packages
WORKDIR /opt/nvidia/deepstream/deepstream/lib
RUN python3 setup.py install
WORKDIR /opt/nvidia/deepstream/deepstream-5.0
RUN pip3 install jupyterlab gdown
COPY English /opt/nvidia/deepstream/deepstream-5.0
WORKDIR /opt/nvidia/deepstream/deepstream-5.0/python/source_code/dataset/
RUN python3 /opt/nvidia/deepstream/deepstream-5.0/python/source_code/dataset/download_dataset.py
RUN ls -l
#RUN mv datasetlihvsi4dtmp  deepstream_dataset.zip
RUN unzip deepstream_dataset.zip
WORKDIR /opt/nvidia/deepstream/deepstream-5.0
## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8889 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-5.0/python

