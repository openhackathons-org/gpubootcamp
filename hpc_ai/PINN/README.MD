# openacc-training-materials
This repository contains mini applications for GPU Bootcamps. The objective of this lab is to give an introduction to application of Artificial Intelligence (AI) algorithms in Science ( High Performance Computing (HPC) Simulations ). This Bootcamp will provide introduction to fundamentals of using Physics Informed Neural Network and how they can be applied to real world scientific domains using NVIDIA Modulus.

## Target Audience:

The target audience for this bootcamp are researchers/graduate students and developers who are new to field of Artifical Intelligence and interested in learning difference between Data and Physics Informed Neural Network approach applied to Simulation domains. Basic Python programming knowledge is required. 

## Tutorial Duration

The overall bootcamp will take approximately 3 hours. There is an additional mini-challenge provided at the end of bootcamp.

## Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run: 
`sudo docker build --network=host -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t myimage:1.0 .`

and to run the container, run:
`sudo docker run --rm -it --gpus=all --network=host -p 8888:8888 myimage:1.0`

The container launches jupyter lab and runs on port 8888
`jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`sudo singularity build <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run <image_name>.simg cp -rT /workspace ~/workspace`


Then, run the container:
`singularity run --nv <image_name>.simg jupyter-lab --notebook-dir=~/workspace/python/jupyter_notebook`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).
