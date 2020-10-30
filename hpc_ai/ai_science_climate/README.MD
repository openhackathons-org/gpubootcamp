# openacc-training-materials
Training materials provided by OpenACC.org. The objective of this lab is to give an introduction to application of Artificial Intelligence (AI) algorithms in Science ( High Performance Computing(HPC) Simulations ). This Bootcamp will introduce you to fundamentals of AI and how they can be applied to Climate/Weather

## Prerequisites:
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/).

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run: 
`sudo docker build --network=host -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t myimage:1.0 .`

and to run the container, run:
`sudo docker run --rm -it --gpus=all --network=host -p 8888:8888 myimage:1.0`

The container launches jupyter notebook and runs on port 8888
`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`sudo singularity build <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run <image_name>.simg cp -rT /workspace ~/workspace`


Then, run the container:
`singularity run --nv <image_name>.simg jupyter notebook --notebook-dir=~/workspace/python/jupyter_notebook`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

