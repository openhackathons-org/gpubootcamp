# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build: $ sudo docker build -t miniprofiler:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 miniprofiler:latest
# Finally, open http://127.0.0.1:8888/

FROM nvcr.io/hpc/pgi-compilers:ce

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get install --no-install-recommends -y \
    openssh-server m4 vim-nox emacs-nox nano zip && \
    rm -rf /var/lib/apt/cache/* 

RUN apt-get update 
RUN apt-get install --no-install-recommends -y python3 python3-pip nginx 
RUN pip3 install --upgrade pip
RUN apt-get install --no-install-recommends -y python3-setuptools
RUN apt-get install --no-install-recommends -y git-core
RUN apt-get install --no-install-recommends -y openssh-client 
RUN apt-get install --no-install-recommends -y inotify-tools 
RUN pip3 install jupyter
RUN pip3 install netcdf4
RUN apt-get install --no-install-recommends -y curl
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install --no-install-recommends -y git-lfs
RUN git lfs install

# NVIDIA Nsight Systems 
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    wget && \
    echo "deb https://developer.download.nvidia.com/devtools/repo-deb/x86_64/ /" >> /etc/apt/sources.list.d/nsight.list && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" >> /etc/apt/sources.list.d/nsight.list  &&\
    apt-get update -y

RUN apt-get update -y

# NVIDIA nsight-systems-2020.1.1 nsight-compute-2020.1.1
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2020.2.1 nsight-compute-2020.1.1

RUN apt-get install --no-install-recommends -y build-essential


ENV PATH="$PATH:/usr/local/bin:/opt/anaconda3/bin:/usr/bin" LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib" CPATH="$CPATH:/usr/local/include"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3  && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/anaconda3/bin/conda install -y -q netcdf4

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/pgi/linux86-64-llvm/2019/cuda/10.1/lib64/" 
ENV PATH="$PATH:/opt/nvidia/nsight-systems/2020.2.1/bin:/opt/nvidia/nsight-compute/2020.1.1/"

ADD English/ /labs
WORKDIR /labs
CMD service nginx start && jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs