# Nsight Tool Tutorial
This repository contains mini applications for GPU Bootcamps (**Tested on NVIDIA driver 440.82**)

## Prerequisites:
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/]).
- Install Nvidia toolkit, [Nsight Systems (latest version)](https://developer.nvidia.com/nsight-systems) and [compute (latest version)](https://developer.nvidia.com/nsight-compute).

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run: 
`sudo docker build -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t myimage:1.0 .`

and to run the container, run:
`sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0`

Then, inside the container launch the Jupyter notebook assigning the port you opened:
`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Once inside the container, start the lab by clicking on the `START_profiling.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`singularity build miniapp_profiler.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run miniapp_profiler.simg cp -rT /labs ~/labs`

Then, run the container:
`singularity run --nv miniapp_profiler.simg jupyter notebook --notebook-dir=~/labs`

Once inside the container, start the lab by clicking on the `START_profiling.ipynb` notebook.

## Questions?
Please join [OpenACC Slack Channel](https://openacclang.slack.com/messages/openaccusergroup) for questions.
