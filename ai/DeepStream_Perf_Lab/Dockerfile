# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# Select Base Image
FROM nvcr.io/nvidia/deepstream:6.0.1-triton


#####
# Read https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
#####

# NVIDIA nsight-systems-cli-2022.1.1, nsight-compute-2022.1.1
RUN apt-get update -y && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
        wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add -  &&\
        echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list &&\
        apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-cli-2022.1.1 nsight-compute-2022.1.1

#Update the repo
RUN apt-get update

#Install required dependencies
#RUN apt-get install ffmpeg python3-gi python3-dev python3-pip cmake unzip -y
RUN apt-get install ffmpeg python3-gi python3-dev python3-gst-1.0 -y
RUN apt-get install unzip

# Install pybind11
RUN pip3 install pybind11
RUN pip3 install gdown


#Install jupyter-lab packages
WORKDIR /opt/nvidia/deepstream/deepstream-6.0
RUN pip3 install jupyterlab
COPY English /opt/nvidia/deepstream/deepstream-6.0

WORKDIR /opt/nvidia/deepstream/deepstream-6.0/python/source_code/dataset/
RUN python3 /opt/nvidia/deepstream/deepstream-6.0/python/source_code/dataset/download_dataset.py
RUN ls -l
#RUN mv datasetlihvsi4dtmp  deepstream_dataset.zip
RUN unzip deepstream_dataset.zip 

#python binding 
WORKDIR /opt/nvidia/deepstream/deepstream/sources 
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/
RUN git submodule update --init
RUN mkdir build

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build
RUN cmake ..  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8
RUN make
RUN apt install libgirepository1.0-dev libcairo2-dev -y
RUN pip3 install ./pyds-1.1.1-py3-none-linux_x86_64.whl


WORKDIR /opt/nvidia/deepstream/deepstream-6.0

## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-6.0/python

