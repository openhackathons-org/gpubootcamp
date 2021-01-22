# RAPIDS_Bootcamp

## GPU Bootcamp for RAPIDS AI

This repository consists of gpu bootcamp material for RAPIDS AI. The RAPIDS suite of open source software libraries gives you the freedom to execute end-to-end data science and analytics pipelines entirely on GPUs. In this series you can access RAPIDS learning resources in the form of labs. The modules covered in this Bootcamp are CuDF, CuML, Dask and Challenge. To access each module individually, you can refer to the respective folders in this repository. To start working on the material head over to the first notebook, Introduction to RAPIDS [here]()

To use RAPIDS in your application, you have several options.<BR>
<BR>
  
# To start the container and notebook server, run the following commands. 
Note that we are mounting the entire tutorial here, but you can mount only your needed codes by altering the -v argument provided.<BR>
  
  ```bash

$ docker pull rapidsai/rapidsai:cuda10.1-runtime-ubuntu18.04-py3.7
$ sudo docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \ 
-v /home/hpclabs/amarathe/MLBootcamp/ai/rapids/English/Python/jupyter_notebook:/rapids/notebooks/host rapidsai/rapidsai:cuda10.1-runtime-ubuntu18.04-py3.7<BR>


```
  
You can open http://localhost:8000 and view the codes there. You can find this tutorial in the host directory. You can also explore different RAPIDs sample notebooks provided in the base directory.

<BR>
  
# To build the DockerContainer from scratch using the Dockerfile provided in this tutorial, run the following commands.<BR>




```bash

# Run from the Root Directory
$ docker build . 

# Run with GPUs and Network Access
$ docker run --gpus all -it --rm -p 8888:8888 ~~~Container ID~~~~

# Run the Jupyter Notebook
$ jupyter notebook --allow-root --ip 0.0.0.0

```
<BR>

# To make your own Dockerfile, and using it to run the tutorial you can follow:<BR>
  
Using the following short Dockerfile users can leverage the existing RAPIDS images and build a custom secure image:<BR>

```bash
FROM rapidsai/rapidsai-nightly:cuda10.2-runtime-ubuntu18.04-py3.7
RUN sed -i "s/NotebookApp.token=''/NotebookApp.token='secure-token-here'/g" /opt/docker/bin/entrypoint_source
```

Once built, the resulting image will be secured with the new token.<BR>

This example can be repurposed by replacing the sed command with other commands for custom libraries or settings.<BR>
  
  
  
  <BR>
## Prerequisites:
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

Make sure both Docker and Singularity has been installed with NVIDIA GPU support

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container, run: 
`sudo docker build --network=host -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build --network=host -t myimage:1.0 .`

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

## Troubleshooting

Q. cuDNN failed to initialize or GPU out of memory error

A. This error occurs when the user forgot to shutdown the jupyter kernel of previously run notebooks. Please make sure that all the previous notebook jupyter kernel is shutdown. ( Go to Home Tab --> Click Running Tab--> Kill notebooks that aren’t being used )

Q. Cannot write to /tmp directory

A. Some notebooks depend on writing logs to /tmp directory. While creating container make sure /tmp director is accesible with write permission to container. Else the user can also change the tmp directory location

Q. "ResourceExhaustedError" error is observed while running the labs
A. Currently the batch size and network model is set to consume 16GB GPU memory. In order to use the labs without any modifications it is recommended to have GPU with minimum 16GB GPU memory. Else the users can play with batch size to reduce the memory footprint

# For more information about RAPIDS applications and Docker, please refer <a href="https://hub.docker.com/r/rapidsai/rapidsai/"> here</a>
