# RIVA_Bootcamp

## GPU Bootcamp for RIVA 

This repository consists of gpu bootcamp material for RIVA. The RIVA frameworks of libraries, APIs and models allows you to build, optimize and deploy voice and speech centric applications and models based entirely on GPUs. In this series you can access RIVA learning resources in the form of labs. The modules covered in this Bootcamp are SpeechToText, Intent Slot Classification, Named Entity Recognition, Question-Answering and Challenge. 

## Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) .

- The base containers required for the lab requires users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

## Pulling containers
To start with, you will have to pull the RIVA Docker container.
ngc registry resource download-version "nvidia/riva/riva_quickstart:1.4.0-beta"

### Copy Docker Initialization Scripts
```cp *.sh riva_quickstart_v1.4.0-beta ```  
```cd riva_quickstart_v1.4.0-beta ```  

### If running with multiple users/clients, please choose a UNIQUE Riva server port number for each user (default is 50051)

### Initialize the RIVA platform
``` bash riva_init_new.sh ```  
``` bash riva_start.sh ```  

### start client

``` bash riva_start_bootcamp.sh```  

### run the Jupyter notebook

``` jupyter notebook --ip=0.0.0.0 --allow-root --notebook-dir=/bootcamp --no-browser --port=8888 --NotebookApp.token='' ```  

Then, open the jupyter notebook in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here_RIVA.ipynb` notebook.


## Troubleshooting

Q. Cannot run RIVA scripts

A. The Riva platform requires Docker to be actively running. In case you are running rootless-docker you should ensure that the Docker daemon is active and has sufficient permissions

Q. Invalid API key

A. Pulling pretrained model containers from NGC requires a valid API key. To get this key, please login to you NGC account and generate a new valid API key and re initialize the Riva platform


## For more information about the Riva platform, models and services, please refer <a href="https://developer.nvidia.com/riva"> here</a>

## For more information about the Riva quickstart container, please refer <a href="https://ngc.nvidia.com/catalog/resources/nvidia:riva:riva_quickstart"> here</a>

