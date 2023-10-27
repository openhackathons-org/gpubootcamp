# Distributed Deep Learning

This folder contains contents for Distributed Deep learning bootcamp.

- Introduction to Distributed deep learning
- Understanding System Topology
- Hands-on with Distributed training ( Horovord, TensorFlow )
- Techniques for faster convergence

## Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

#Tutorial Duration
The total bootcamp material would take approximately 5 hours ( including solving mini-challenge ).

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run:
`sudo docker build --network=host -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t myimage:1.0 .`

The code labs have been written using Jupyter notebooks and a Dockerfile has been built to simplify deployment. In order to serve the docker instance for a student, it is necessary to expose port 8888 from the container, for instance, the following command would expose port 8888 inside the container as port 8888 on the lab machine:

`sudo docker run --rm -it --gpus=all -p 8888:8888 -p 8000:8000 myimage:1.0`

When this command is run, you can browse to the serving machine on port 8888 using any web browser to access the labs. For instance, from if they are running on the local machine the web browser should be pointed to http://localhost:8888. The `--gpus` flag is used to enable `all` NVIDIA GPUs during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. This command may be customized for your hosting environment.


Once inside the container launch the jupyter notebook by typing the following command
`jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/jupyter_notebook/`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run:
`sudo singularity build --sandbox <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run --writable <image_name>.simg cp -rT /workspace/ ~/workspace`


Then, run the container:
`singularity run --nv --writable <image_name>.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/jupyter_notebook/`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Known Issues

Q. "ResourceExhaustedError" error is observed while running the labs
A. Currently the batch size and network model is set to consume 16GB GPU memory. In order to use the labs without any modifications it is recommended to have GPU with minimum 16GB GPU memory. Else the users can play with batch size to reduce the memory footprint

- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).
