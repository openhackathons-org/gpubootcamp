# openacc-training-materials

This repository contains mini applications for GPU Bootcamps (**Tested on NVIDIA driver 440.82**). This bootcamp covers how to program GPUs with OpenACC though hands on experience.

**Tutorial Outline**
- Lab 1: Introduction to OpenACC 
- Lab 2: OpenACC Data Management 
- Lab 3: Loop Optimizations with OpenACC

## Target Audience

The target audience for this lab is researchers/graduate students and developers who are interested in learning about programming GPUs with OpenACC.

## Tutorial Duration

The total bootcamp material  would take approximately 3 hours (  1 hour per Lab ).

## Prerequisites:
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/]).
- Install Nvidia toolkit, [Nsight Systems (latest version)](https://developer.nvidia.com/nsight-systems) and [compute (latest version)](https://developer.nvidia.com/nsight-compute).
- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account).

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run: 
`sudo docker build -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t myimage:1.0 .`

The code labs have been written using Jupyter lab and a Dockerfile has been built to simplify deployment. In order to serve the docker instance for a student, it is necessary to expose port 8000 from the container, for instance, the following command would expose port 8000 inside the container as port 8000 on the lab machine:

`sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0`

When this command is run, you can browse to the serving machine on port 8000 using any web browser to access the labs. For instance, from if they are running on the local machine the web browser should be pointed to http://localhost:8000. The `--gpus` flag is used to enable `all` NVIDIA GPUs during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. This command may be customized for your hosting environment.

<!---
Then, inside the container launch the Jupyter lab assigning the port you opened:

`jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`
-->

Once inside the container, open the jupyter lab in browser: http://localhost:8888, and start the lab by clicking on the `START_profiling.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`sudo singularity build openacc.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run openacc.simg cp -rT /labs ~/labs`

Then, run the container:
`singularity run --nv openacc.simg jupyter-lab  --notebook-dir=~/labs`

Once inside the container, open the jupyter lab in browser: http://localhost:8888, and start the lab by clicking on the `START_profiling.ipynb` notebook.



## Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).
