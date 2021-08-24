# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

Bootstrap: docker
FROM: nvcr.io/nvidia/nvhpc:21.5-devel-cuda_multi-ubuntu20.04

%environment
    export XDG_RUNTIME_DIR=
    export PATH="/opt/openmpi/ompi/bin/:/usr/local/bin:/opt/anaconda3/bin:/usr/bin:/opt/nvidia/nsight-systems/2020.5.1/bin:/opt/nvidia/nsight-compute/2020.2.1:$PATH"
    export LD_LIBRARY_PATH="/opt/openmpi/ompi/lib:/pmi_utils/lib/:/usr/local/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/lib64/:$LD_LIBRARY_PATH"

%post
    build_tmp=$(mktemp -d) && cd ${build_tmp}

    apt-get -y update
    apt-get -y dist-upgrade 
    DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends \
	    m4 vim-nox emacs-nox nano zip\
 	    python3-pip python3-setuptools git-core inotify-tools \
	    curl git-lfs \
	    build-essential libtbb-dev
    rm -rf /var/lib/apt/cache/* 

    pip3 install --upgrade pip
    pip3 install --no-cache-dir jupyter
    pip3 install --no-cache-dir jupyterlab
    pip3 install gdown

    apt-get install --no-install-recommends -y build-essential 

# NVIDIA nsight-systems-2020.5.1 ,nsight-compute-2
    apt-get update -y   
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg wget
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F60F4B3D7FA2AF80
    echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list 
    apt-get update -y 
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2020.5.1 nsight-compute-2020.2.1 
    apt-get install --no-install-recommends -y build-essential

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3 
    rm Miniconda3-latest-Linux-x86_64.sh 

# Install CUDA-aware OpenMPI with UCX and PMI
    mkdir -p /opt/openmpi && cd /opt/openmpi
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
    tar -xvzf openmpi-4.1.1.tar.gz
    mkdir -p /opt/openmpi/ompi/
    cd /opt/openmpi/openmpi-4.1.1/
    ./configure --prefix=/opt/openmpi/ompi/ --with-libevent=internal --with-xpmem --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/ --with-slurm --with-pmix=internal --with-pmi=/pmi_utils/ --enable-mpi1-compatibility --with-verbs --with-hcoll=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/comm_libs/hpcx/hpcx-2.8.1/hcoll/ --with-ucx=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/comm_libs/hpcx/hpcx-2.8.1/ucx/
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/pmi_utils/lib/"
    make all install
    
    cd /
    rm -rf ${build_tmp}

%files
    labs/ /labs
    slurm_pmi_config/ /pmi_utils

%runscript
    "$@"

%labels
    AUTHOR Anish-Saxena
