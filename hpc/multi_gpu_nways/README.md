# N-Ways to Multi-GPU Programming

This bootcamp focuses on multi-GPU programming models. 

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
* Final remarks

**NOTE:** NCCL, NVSHMEM, and Final Remarks notebooks are work under progress. All other notebooks are available.

## Prerequisites

This bootcamp requires a multi-node system with multiple GPUs in each node (atleast 2 GPUs/ node). 

A multi-node installation of [NVIDIA's HPC SDK](https://developer.nvidia.com/hpc-sdk) is desired. 

Otherwise, multi-node compatible versions of the following are required:

* [Singularity](https://sylabs.io/docs/%5D)
* [OpenMPI](https://www.open-mpi.org/)
* [HPCX](https://developer.nvidia.com/networking/hpc-x)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [NCCL](https://developer.nvidia.com/nccl)

## Testing

We have tested all the codes with CUDA drivers 460.32.03 with CUDA 11.3.0.0, OpenMPI 4.1.1, HPCX 2.8.1, Singularity 3.6.1, and NCCL 2.9.9.1. 

### Containerless Build

As this bootcamp covers multi-node CUDA-aware MPI concepts, it is primarily designed to run without any containers. After the prerequisite softwares have been installed, follow these steps to install Jupyter Lab:

```bash
# Install Anaconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh -b -p <my_dir>
# Add conda to PATH
export PATH=$PATH:<my_dir>/bin/
# Install Jupyter Lab
conda install -c conda-forge jupyterlab
```

### Containerized Build:

If containerization is desired, follow the steps outlined in the notebook [MPI in Containerized Environments](labs/CFD/English/C/jupyter_notebook/mpi/containers_and_mpi.ipynb).

#### Building Singularity Container

```bash
singularity build multi_gpu_nways.simg Singularity
```

### Running Jupyter Lab

#### Containerless Build

```bash
jupyter lab --notebook-dir=./labs/ --port=8000 --ip=0.0.0.0 --no-browser --NotebookApp.token=""
```

#### Containerized Build

```bash
singularity run --nv multi_gpu_nways.simg jupyter lab --notebook-dir=./labs/ --port=8000 --ip=0.0.0.0 --no-browser --NotebookApp.token="" 
```

### Accessing Bootcamp

After running Jupyter Lab, open [http://localhost:8888](http://localhost:8888/) in a web browser and start the `introduction.ipynb` notebook.

## Questions?

Please join [OpenACC Slack Channel](https://openacclang.slack.com/messages/openaccusergroup) to raise questions.

If you observe any errors or issues, please file an issue on [GPUBootcamp GitHuB repository](https://github.com/Anish-Saxena/gpubootcamp/tree/hpc-multi-gpu).
