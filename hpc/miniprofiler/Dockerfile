# To build: $ sudo docker build -t myimage:1.0 .

# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0

# To run Jupyter inside the container: $ jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

FROM nvcr.io/nvidia/cuda:10.2-base-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    openssh-server m4 vim-nox emacs-nox nano zip wget python3 python3-pip python3-setuptools git-core inotify-tools curl git-lfs  build-essential && \
    rm -rf /var/lib/apt/cache/* 

RUN pip3 install --upgrade pip
RUN apt-get install --no-install-recommends -y openssh-client 
RUN pip3 install jupyter
RUN pip3 install netcdf4

### NVIDIA HPC SDK 20.7
RUN wget https://developer.download.nvidia.com/hpc-sdk/nvhpc_2020_207_Linux_x86_64_cuda_multi.tar.gz
RUN tar xpzf nvhpc_2020_207_Linux_x86_64_cuda_multi.tar.gz

RUN nvhpc_2020_207_Linux_x86_64_cuda_multi/install
###

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3  && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/anaconda3/bin/conda install -y -q netcdf4

ENV PATH="$PATH:/opt/pgi/linux86-64/19.10/bin:/usr/local/bin:/opt/anaconda3/bin:/usr/bin" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib" CPATH="$CPATH:/usr/local/include"

ENV NVARCH=`uname -s`_`uname -m`; export NVARCH \
    NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS \
    MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/20.7/compilers/man; export MANPATH

ENV PATH=$NVCOMPILERS/$NVARCH/20.7/compilers/bin:$PATH; export PATH \
    PATH=$NVCOMPILERS/$NVARCH/20.7/comm_libs/mpi/bin:$PATH \
    MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/20.7/comm_libs/mpi/man

ADD English/ /labs