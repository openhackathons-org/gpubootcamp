# To build: $ sudo docker build -t myimage:1.0 .

# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0

# To run Jupyter inside the container: $ jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

FROM nvcr.io/hpc/pgi-compilers:ce

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install --no-install-recommends -y \
    openssh-server m4 vim-nox emacs-nox nano zip && \
    rm -rf /var/lib/apt/cache/* 
#useradd -k /etc/skel -m -s /usr/local/bin/entrypoint.sh -p openacc openacc && \
#echo 'openacc:openacc' | chpasswd && \
#mkdir /var/run/sshd 

RUN apt-get install --no-install-recommends -y python3 python3-pip
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
    apt-get update -y

RUN apt-get update -y

# NVIDIA Nsight Systems 2020.3.1
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2020.3.1


#RUN apt-get update && apt-get install -y --no-install-recommends && \
#    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list 

# NVIDIA Nsight Systems 2019.3.7
#RUN apt-get update -y && \ 
#    apt-get install -y cuda-nsight-systems-10-1 nsight-systems-2019.3.7

RUN apt-get install --no-install-recommends -y build-essential

##### START : netcdf installation #####

RUN cd /usr && \
    mkdir netcdf && \
    cd netcdf && \
    mkdir WORK_DIR && \
    cd /usr/netcdf/WORK_DIR && \
    mkdir zlib hdf5 netCDF netCDF-C++ netCDF-Fortran

RUN cd /usr/netcdf/WORK_DIR/zlib && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz && \
    tar -xvzf zlib-1.2.8.tar.gz

RUN cd /usr/netcdf/WORK_DIR/hdf5 && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.12.tar.gz && \
    tar -xvzf hdf5-1.8.12.tar.gz

RUN cd /usr/netcdf/WORK_DIR/netCDF && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/old/netcdf-4.3.0.tar.gz && \
    tar -xvzf netcdf-4.3.0.tar.gz 

RUN cd /usr/netcdf/WORK_DIR/netCDF-C++ && \
    wget https://github.com/Unidata/netcdf-cxx4/archive/v4.2.1.tar.gz && \
    tar -xvzf v4.2.1.tar.gz

RUN cd /usr/netcdf/WORK_DIR/netCDF-Fortran && \
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/old/netcdf-fortran-4.2.tar.gz && \
    tar -xvzf netcdf-fortran-4.2.tar.gz

ENV CC=pgcc CFLAGS="-O tp=p7-64" CXX=pgc++ CXXFLAGS="-O tp=p7-64" FC=pgfortran FCFLAGS="-O tp=p7-64" F77=pgfortran FFLAGS="-O tp=p7-64" CPPFLAGS="-DpgiFortran" 

RUN mkdir -p /usr/local && \
    mkdir -p /usr/local/bin && \
    mkdir -p /usr/local/include && \
    mkdir -p /usr/local/lib 

ENV PATH="$PATH:/usr/local/bin:/opt/anaconda3/bin:/usr/bin" LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib" CPATH="$CPATH:/usr/local/include"

# zlib 
RUN cd /usr/netcdf/WORK_DIR/zlib/zlib-1.2.8 && \
    ./configure --prefix=/usr/local && \
    make  && \
    make install 

# hdf5
RUN cd  /usr/netcdf/WORK_DIR/hdf5/hdf5-1.8.12 && \
    unset CPP && \
    ./configure --prefix=/usr/local --enable-fortran --enable-c++ && \
    make  && \
    make install 

# netcdf
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" && \
    export CFLAGS="-O tp=p7-64 -I /usr/local/include" && \
    export LDFLAGS="-L/usr/local/lib -L/usr/local/lib" && \
    cd  /usr/netcdf/WORK_DIR/netCDF/netcdf-4.3.0 && \
    ./configure --prefix=/usr/local && \
    make  && \
    make install  

# netcdf-c++
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/hdf5/lib:/usr/local/lib" && \
    export CFLAGS="-O tp=p7-64 -I /usr/local/include -I /usr/local/include" && \
    export CPPFLAGS="-DpgiFortran -I /usr/local/include -I /usr/local/include" && \
    export LDFLAGS="-L/usr/local/lib -L/usr/local/lib" && \
    cd  /usr/netcdf/WORK_DIR/netCDF-C++/netcdf-cxx4-4.2.1 && \
    ./configure --prefix=/usr/local && \
    make  && \
    make install 

# netcdf-fortran
RUN export CFLAGS="-O tp=p7-64 -I /usr/local/include -I /usr/local/include" && \
    export FCFLAGS="-O tp=p7-64 -I /usr/local/include -I /usr/local/include" && \
    export FFLAGS="-O tp=p7-64 -I /usr/local/include -I /usr/local/include" && \
    export CPPFLAGS="-DpgiFortran -I /usr/local/include -I /usr/local/include" && \
    unset LDFLAGS && \
    cd  /usr/netcdf/WORK_DIR/netCDF-Fortran/netcdf-fortran-4.2 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install   
##### END : netcdf installation #####

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3  && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/anaconda3/bin/conda install -y -q netcdf4

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/pgi/linux86-64-llvm/2019/cuda/10.1/lib64/" 

ADD miniapps-profiler /labs