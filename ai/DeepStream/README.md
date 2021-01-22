
# openacc-training-materials
Training materials provided by OpenACC.org. The objective of this lab is to give an introduction to using Nvidia DeepStream Framework in a Intelligent Video Analytics Domain.  

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

Once inside the container launch the jupyter notebook by typing the following command
`jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-5.0/python`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run:
`sudo singularity build <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run <image_name>.simg cp -rT /workspace ~/workspace`


Then, run the container:
`singularity run --nv <image_name>.simg jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=~workspace/python`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Troubleshooting

Q. "ResourceExhaustedError" error is observed while running the labs
A. Currently the batch size and network model is set to consume 16GB GPU memory. In order to use the labs without any modifications it is recommended to have GPU with minimum 16GB GPU memory. Else the users can play with batch size to reduce the memory footprint

