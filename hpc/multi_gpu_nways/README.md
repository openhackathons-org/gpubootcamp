# N-Ways to Multi-GPU Programming

This repository contains mini applications for GPU Bootcamps. This bootcamp focuses on multi-GPU programming models.

Scaling applications to multiple GPUs across multiple nodes requires one to be adept at not just the programming models and optimization techniques, but also at performing root-cause analysis using in-depth profiling to identify and minimize bottlenecks. In this bootcamp, participants will learn to improve the performance of an application step-by-step, taking cues from profilers along the way. Moreover, understanding of the underlying technologies and communication topology will help us utilize high-performance NVIDIA libraries to extract more performance out of the system.

## Bootcamp Outline

* Overview of single-GPU code and Nsight Systems Profiler
* Single Node Multi-GPU:
  - CUDA Memcpy and Peer-to-Peer Memory Access
  - Intra-node topology
  - CUDA Streams and Events
* Multi-Node Multi-GPU:
  - Introduction to MPI and Multi-Node execution overview
  - MPI with CUDA Memcpy
  - CUDA-aware MPI
  - Supplemental: Configuring MPI in a containerized environment
* NVIDIA Collectives Communications Library (NCCL)
* NVHSMEM Library

## Prerequisites

This bootcamp requires a multi-node system with multiple GPUs in each node (atleast 2 GPUs/ node).

## Tutorial Duration

The total bootcamp material  would take approximately 8 hours .

### Using NVIDIA HPC SDK

A multi-node installation of [NVIDIA's HPC SDK](https://developer.nvidia.com/hpc-sdk) is desired. Refer to [NVIDIA HPC SDK Installation Guide](https://docs.nvidia.com/hpc-sdk/hpc-sdk-install-guide/index.html) for detailed instructions. Ensure that your installation contains HPCX with UCX. 

After installation, make sure to add HPC SDK to the environment as follows(For example the PATH highlighted below is for HPC SDK 21.5):

```bash
# Add HPC-SDK to PATH:
export PATH="<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/compilers/bin:<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/cuda/bin:$PATH"
# Add HPC-SDK to LD_LIBRARY_PATH:
export LD_LIBRARY_PATH="<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/comm_libs/nvshmem/lib:<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/comm_libs/nccl/lib:<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/comm_libs/mpi/lib:<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/math_libs/lib64:<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/compilers/lib:<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/cuda/extras/CUPTI/lib64:<path-nvidia-hpc-sdk>>/Linux_x86_64/21.5/cuda/lib64:$LD_LIBRARY_PATH"
#ADD NVSHMEM HOME DIRECTORY PATH
export CUDA_HOME=<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/cuda
export NVSHMEM_HOME=<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/comm_libs/nvshmem
```
**Note:** If you don't use Slurm workload manager, remove `--with-slurm` flag.

Then, install OpenMPI as follows:

```bash
# Download and extract OpenMPI Tarfile
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
tar -xvzf openmpi-4.1.1.tar.gz
cd openmpi-4.1.1/
mkdir -p build
# Configure OpenMPI
./configure --prefix=$PWD/build --with-libevent=internal --with-xpmem --with-cuda=<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/cuda/ --with-slurm --enable-mpi1-compatibility --with-verbs --with-hcoll=<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/comm_libs/hpcx/hpcx-2.8.1/hcoll/lib --with-ucx=<path-to-nvidia-hpc-sdk>/Linux_x86_64/21.5/comm_libs/hpcx/hpcx-2.8.1/ucx/
# Install OpenMPI
make all install
```

Now, add OpenMPI to the environment:

```bash
export PATH="<path-to-openmpi>/build/bin/:$PATH"
export LD_LIBRARY_PATH="<path-to-openmpi/build/lib:$LD_LIBRARY_PATH"
```

Ensure that the custom-built OpenMPI is in use by running `which mpirun` which should point the `mpirun` binary in `<path-to-openmpi>/build/bin` directory.

### Without Using NVIDIA HPC SDK

Multi-node compatible versions of the following are required:

* [OpenMPI](https://www.open-mpi.org/)
* [HPCX](https://developer.nvidia.com/networking/hpc-x)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [NCCL](https://developer.nvidia.com/nccl)
* [NVSHMEM](https://developer.nvidia.com/nvshmem)

## Testing

We have tested all the codes with CUDA drivers 460.32.03 with CUDA 11.3.0.0, OpenMPI 4.1.1, HPCX 2.8.1, Singularity 3.6.1, NCCL 2.9.9.1, and NVSHMEM 2.1.2. Note that OpenMPI in our cluster was compiled with CUDA, HCOLL, and UCX support.

## Running Jupyter Lab

As this bootcamp covers multi-node CUDA-aware MPI concepts, it is primarily designed to run without any containers. After the prerequisite softwares have been installed, follow these steps to install and run Jupyter Lab:

```bash
# Install Anaconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh -b -p <my_dir>
# Add conda to PATH
export PATH=$PATH:<my_dir>/bin/
# Install Jupyter Lab
conda install -c conda-forge jupyterlab
# Run Jupyter Lab
jupyter lab --notebook-dir=<path-to-gpubootcamp-repo>/hpc/multi_gpu_nways/labs/ --port=8000 --ip=0.0.0.0 --no-browser --NotebookApp.token=""
```

After running Jupyter Lab, open [http://localhost:8888](http://localhost:8888/) in a web browser and start the `introduction.ipynb` notebook.

## Optional: Containerized Build with Singularity

This material is designed to primarily run in containerless environments, that is, directly on the cluster. Thus, building the Singularity container is OPTIONAL.

If containerization is desired, follow the steps outlined in the notebook [MPI in Containerized Environments](labs/CFD/English/C/jupyter_notebook/mpi/containers_and_mpi.ipynb).

Follow the steps below to build the Singularity container image and run Jupyter Lab:

```bash
# Build the container
singularity build multi_gpu_nways.simg Singularity
# Run Jupyter Lab
singularity run --nv multi_gpu_nways.simg jupyter lab --notebook-dir=<path-to-gpubootcamp-repo>/hpc/multi_gpu_nways/labs/ --port=8000 --ip=0.0.0.0 --no-browser --NotebookApp.token="" 
```

Then, access Jupyter Lab on [http://localhost:8888](http://localhost:8888/).


## Known issues

#### Compiler throws errors

If compiling any program throws an error related to CUDA/ NCCL/ NVHSMEM/ MPI libraries or header files being not found, ensure that `LD_LIBRARY_PATH` is correctly set. Moreover, make sure environment variables `CUDA_HOME`, `NCCL_HOME`, and `NVSHMEM_HOME` are set either during installation or manually inside each `Makefile`.

- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).


## Questions?

Please join [OpenACC Slack Channel](https://openacclang.slack.com/messages/openaccusergroup) to raise questions.

If you observe any errors or issues, please file an issue on [GPUBootcamp GitHuB repository](https://github.com/gpuhackathons-org/gpubootcamp).
