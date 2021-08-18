# Nsight Tool Tutorial
This repository contains mini applications for GPU Bootcamps (**Tested on NVIDIA driver 440.82**). In this bootcamp, we will be optimizing the serial Weather Simulation application written in both C and Fortran programming language

- Introduction: Overview of profiling tools and Mini Weather application
- Lab 1: Profile Serial application to find hotspots using NVIDIA Nsight System
- Lab 2: Parallelise the serial application using OpenACC compute directives
- Lab 3: OpenACC optimization techniques
- Lab 4: Apply incremental parallelization strategies and use profiler's report for the next step
- Lab 5: Nsight Compute Kernel Level Analysis ( Optional )

## Target Audience

The target audience for this bootcamp are researchers/graduate students and developers who are interested in getting hands on experience with the NVIDIA Nsight System through profiling a real life parallel application using OpenACC programming model and NVTX.

## Tutorial Duration

The bootcamp material would take approximately 2 hours. Link to material is available for download at the end of the lab.


## Prerequisites:
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/]).
- Install Nvidia toolkit, [Nsight Systems (latest version)](https://developer.nvidia.com/nsight-systems) and [compute (latest version)](https://developer.nvidia.com/nsight-compute).
- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run: 
`sudo docker build -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t myimage:1.0 .`

The code labs have been written using Jupyter notebooks and a Dockerfile has been built to simplify deployment. In order to serve the docker instance for a student, it is necessary to expose port 8000 from the container, for instance, the following command would expose port 8000 inside the container as port 8000 on the lab machine:

`sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0`

When this command is run, you can browse to the serving machine on port 8000 using any web browser to access the labs. For instance, from if they are running on the local machine the web browser should be pointed to http://localhost:8000. The `--gpus` flag is used to enable `all` NVIDIA GPUs during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. This command may be customized for your hosting environment.


Then, inside the container launch the Jupyter notebook assigning the port you opened:

`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`


Once inside the container, open the jupyter notebook in browser: http://localhost:8888, and start the lab by clicking on the `START_profiling.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`sudo singularity build miniapp_profiler.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run miniapp_profiler.simg cp -rT /labs ~/labs`

Then, run the container:
`singularity run --nv miniapp_profiler.simg jupyter notebook --notebook-dir=~/labs`

Once inside the container, open the jupyter notebook in browser: http://localhost:8888, and start the lab by clicking on the `START_profiling.ipynb` notebook.


## Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).
