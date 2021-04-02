
# openacc-training-materials
Training materials provided by OpenACC.org. The objective of this lab is to provide insight into DeepStream performance optimization cycle. The lab will make use of Nvidia Nsight System for profiling Nvidia DeepStream pipeline in a Intelligent Video Analytics Domain.  

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

The container launches jupyter notebook and runs on port 8888
`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Once inside the container launch the jupyter notebook by typing the following command
`jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-5.0/python`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run:
`sudo singularity build --sandbox <image_name>.simg Singularity`

and copy the files to your local machine to make sure changes are stored locally:
`singularity run --writable <image_name>.simg cp -rT /opt/nvidia/deepstream/deepstream-5.0/ ~/workspace`


Then, run the container:
`singularity run --nv --writable <image_name>.simg jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=~/workspace/python`

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Troubleshooting

