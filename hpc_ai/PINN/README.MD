# openacc-training-materials
This repository contains mini applications for GPU Bootcamps. The objective of this lab is to give an introduction to application of Artificial Intelligence (AI) algorithms in Science ( High Performance Computing (HPC) Simulations ). This Bootcamp will provide introduction to fundamentals of using Physics Informed Neural Network and how they can be applied to real world scientific domains using NVIDIA SimNet.

- Introduction: Data Driven vs PINN approach
- Lab 1: Solving Partial Differential Equations using SimNet
- Lab 2: Solving transient problems and inverse problems using SimNet
- Lab 3: Mini Challenge

## Target Audience:

The target audience for this bootcamp are researchers/graduate students and developers who are new to field of Artifical Intelligence and interested in learning difference between Data and Physics Informed Neural Network approach applied to Simulation domains. Basic Python programming knowledge is required. 

## Tutorial Duration

The overall bootcamp will take approximately 3 hours. There is an additional mini-challenge provided at the end of bootcamp.

## Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab needs to be downloaded from SimNet product page and requires users to create a Nvidia Developer account (https://developer.nvidia.com/nvidia-simnet-downloads)

## Creating containers
To start with, we need to download the Docker file (`simnet_image_<version>.tar.gz`)as mentioned in the Prerequisites section. (Please note this lab is tested with SimNet release version 21.06). 

### Docker Container
Load the docker tar image: 
`docker load -i simnet_image_v21.06.tar.gz`

and to run the container, run:
`docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v ${PWD}/English/python:/examples -it simnet:21.06 jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/examples`

The container launches jupyter notebook and runs on port 8888
`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`sudo singularity build <image_name>.simg simnet_image_v21.06.tar.gz`


Then, run the container:
`singularity run --nv <image_name>.simg i--bind ${PWD}/English/python:/examples jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/examples`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).
