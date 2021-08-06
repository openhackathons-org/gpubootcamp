# openacc-training-materials
Training materials provided by OpenACC.org. The objective of this lab is to give an introduction to application of Artificial Intelligence (AI) algorithms in Science ( High Performance Computing (HPC) Simulations ). This Bootcamp will provide introduction to fundamentals of using Physics Informed Neural Network and how they can be applied to real world scientific domains using NVIDIA SimNet. 

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

