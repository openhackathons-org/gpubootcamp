# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $ sudo docker build -t openacc-labs:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 openacc-labs:latest
# Finally, open http://127.0.0.1:8888/

FROM nvcr.io/nvidia/nvhpc:20.9-devel-ubuntu20.04

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip
    
RUN apt-get update -y        
RUN apt-get install -y git nvidia-modprobe
RUN pip3 install jupyterlab
# Install required python packages
RUN pip3 install ipywidgets


ADD English/ /labs
WORKDIR /labs
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs
