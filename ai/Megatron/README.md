# Practical Guide to Train Megatron-LM with your own language

This folder contains contents for Practical Guide to train Megatron-LM GPT Model with your own language.
There are 2 Labs, each with a different focus. 

- **Outlines of Lab 1**
 Megatron-LM 101 in half a day - a walk through of Megatron-LM's default workflow.

- **Outlines of Lab 2**
 Customizing Megatron-LM's workflow to adjust to local language needs.

**Important** : This boot camp is intended to be delivered by NVIDIA certified instructors and TAs, it is _NOT_ meant for self-paced learners.

Note1 : The lecture presentations as well as the solutions to the challenges and mini-challenges will be delivered at the end of each lab
Note2 : Multi-nodes Megatron-LM GPT3 training can be added as an additional lab depending on the availability of the compute resource.

## Labs Duration :
The two labs will take approximately 12 hours (including solving challenges and mini-challenges) to complete.

## Compute Resources & Environment preparation:

Although this boot camp is designed to run on a computing cluster with [NVIDIA SuperPOD Architecture](https://resources.nvidia.com/en-us-auto-datacenter/nvpod-superpod-wp-09)
It is possible to run it in an environment where you have access to 2 X A100 GPUs 40 GB with NVLink/Switch.

### Scenario 1 : local station with 2 X A100 GPU 40 GB and NVLINK 
When docker pull & run is allowed, and the GPUs are directly accessbile to the users in the environment.

#### Step 1 - Clone the gpubootcamp repo to obtain the scripts and notebooks.
`git clone https://github.com/gpuhackathons-org/gpubootcamp.git &&
cd gpubootcamp `

#### Step 2 - run PyTorch docker image 
USR_PORT=
GPUS= # you only need two gpus
DIR= 
With sudo privilege : 
`sudo docker run --runtime=NVIDIA -e NVIDIA_VISIBLE_DEVICES=GPUS -p USR_PORT:USR_PORT -it --rm --ulimit hemlock=-1 --ulimit stack=67108864 --cap-add=SYS_ADMIN -v DIR:/workspace nvcr.io/nvidia/pytorch:21.03-py3 `

Without sudo privilege but the user is added to the docker group : 
`docker run --runtime=NVIDIA -e NVIDIA_VISIBLE_DEVICES=GPUS -p USR_PORT:USR_PORT -it --rm --ulimit hemlock=-1 --ulimit stack=67108864 -v DIR:/workspace nvcr.io/nvidia/pytorch:21.03-py3`

#### Step 3 - call out Jupiter lab 
` Jupiter lab --no-browser --port=USR_PORT --allow-root --NotebookApp.token='' `

#### Step 4 - in another terminal , call out a browser (such as firefox)
Then, open the Jupiter notebook in browser: local host:USR_PORT
Navigate to /gpubootcamp/ai/Megatron/English/Python/ and open the `Start_Here.ipynb` notebook.

### Scenario 2 : Accessing the Jupiter lab with Singularity + Slurm + SSH port forwarding is allowed
A User Guide is often provided when one requests for access to a computing cluster with [NVIDIA SuperPOD Architecture](https://resources.nvidia.com/en-us-auto-datacenter/nvpod-superpod-wp-09). However, each compute cluster might have slight deviations to the reference architecture on various levels, HW and/or SW as well as the resource management control setups. 

It is likely the below steps will need to be adjusted, in which case, the user will need to consult the cluster admin or cluster operator to get help in debugging environment preparation in order to prepare for the boot camp materiel to run.

#### Step 1 - Clone the gpubootcamp repo to obtain the scripts and notebooks.
` clone https://github.com/gpuhackathons-org/gpubootcamp.git`

DIR_to_gpubootcamp=
USR_PORT= # communicate with the cluster admin to know which port is available to you as a user
HOST_PORT=
CLUSTER_NAME=
#### Step 2 - Build the PyTorch_21.03.sif file 
`sudo singularity build PyTorch_21.03.sif docker://nvcr.io/pytorch:21.03-py3`

Note1: If you do not have sudo rights, you might need to either contact the cluster admin, or build this in another environment where you have sudo rights.
Note2: You should copy the PyTorch_21.03.sif to the cluster environment one level above the DIR_to_gpubootcamp

#### Step 3 - request 2 A100 GPUs resource 
`srun --gres=gpu:2 --pty bash -i`

#### Step 4 - request 2 A100 GPUs resource 
`export SINGULARITY_BINDPATH=DIR_to_gpubootcamp`

#### Step 5 - Run singularity with the PyTorch_21.03.sif file 
`singularity run --nv PyTorch_21.03.sif Jupiter lab --notebook-dir=DIR_to_gpubootcamp --port=USR_PORT --ip=0.0.0.0 --no-browser --NotebookApp.iopub_data_rate_limit=1.0e15 --NotebookApp.token="" 
`
#### Step 6 - ssh and Port forwarding 
` ssh -L local host:HOST_PORT:machine_number:USR_PORT CLUSTER_NAME`

#### Step 7 - in another terminal , call out a browser (such as firefox)
Then, open the Jupiter notebook in browser: local host:HOST_PORT
Navigate to gpubootcamp/ai/Megatron/English/Python/ and open the `Start_Here.ipynb` notebook.


## Known Issues

Q. "ResourceExhaustedError" error is observed while running the labs
A. Currently the batch size and network model is set to consume 40 GB GPU memory. In order to use the labs without any modifications it is recommended to have GPU with minimum 40 GB GPU memory. Else the users can play with batch size to reduce the memory footprint, also ensure you have NVLINK/Switch enabled in the environment.Do not enable MIG mode when requesting A100 GPUs as resources.

- Please go through the list of existing bugs/issues or file a new issue at [Github](https://github.com/gpuhackathons-org/gpubootcamp/issues).

