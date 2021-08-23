
# openacc-training-materials
This repository contains mini applications for GPU Bootcamps. The objective of this Bootcamp is to give an introduction to using NVIDIA DeepStream Framework and apply to Intelligent Video Analytics Domain.  

- Introduction to Deepstream and Gstreamer
- Lab 1: Getting started with Deepstream Pipeline
- Lab 2: Introduction to Multi-DNN pipeline
- Lab 3: Creationg multi-stream pipeline
- Lab 4: Mini-Challenge : Combining Multi-stream with Multi-DNN pipeline

## Target Audience:

The target audience for this bootcamp are AI developers working in domain of Intelligent Video Anaytics and looking at optimizing the application using NVIDIA DeepStream SDK.

## Tutorial Duration

The overall lab should take approximate 3.5 hours. There is an additional mini-challenge provided at the end of lab.  

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

Once inside the container launch the jupyter lab by typing the following command
`jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-5.0/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run:
`sudo singularity build --sandbox <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run --writable <image_name>.simg cp -rT /opt/nvidia/deepstream/deepstream-5.0/ ~/workspace`


Then, run the container:
`singularity run --nv --writable <image_name>.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=~/workspace/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Known issues

Q. "ResourceExhaustedError" error is observed while running the labs

A. Currently the batch size and network model is set to consume 16GB GPU memory. In order to use the labs without any modifications it is recommended to have GPU with minimum 16GB GPU memory. Else the users can play with batch size to reduce the memory footprint

- If you observe any errors, please file an issue on [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).
