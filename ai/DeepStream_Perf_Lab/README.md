
# openacc-training-materials
This repository contains mini applications for GPU Bootcamps. The objective of this Bootcamp is to provide insight into DeepStream performance optimization cycle. The lab will make use of NVIDIA Nsight System for profiling Nvidia DeepStream pipeline in a Intelligent Video Analytics Domain.  

- Introduction: Performance analysis
- Lab 1: Performance Analysis using NVIDIA Nsight systems
- Lab 2: COVID-19 Social Distancing Application plugin optimization

## Target Audience:

The target audience for this bootcamp are NVIDIA DeepStream users and looking at understanding performance optimization cycle using profilers. Users are recommended to go through basic of [DeepStream SDK](https://github.com/gpuhackathons-org/gpubootcamp/tree/master/ai/DeepStream) if not already done. 

## Tutorial Duration

The overall lab should take approximate 3.5 hours.


## Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- Install Nvidia toolkit, [Nsight Systems (latest version)](https://developer.nvidia.com/nsight-systems).

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

Once inside the container launch the jupyter lab by typing the following command
`jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-6.0/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run:
`sudo singularity build --sandbox <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run --writable <image_name>.simg cp -rT /opt ~/opt`


Then, run the container:
`singularity run --nv --writable <image_name>.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=~/opt/nvidia/deepstream/deepstream-6.0/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).


