# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t ai-science-climate:latest --network=host .
# To run: $ sudo docker run --rm -it --gpus=all --network=host -p 8888:8888 ai-science-climate:latest
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3
# Update the repo
RUN apt-get update -y
# Install required dependencies
RUN apt-get install -y libsm6 libxext6 libxrender-dev git nvidia-modprobe
# Install required python packages
RUN pip3 install  opencv-python==4.1.2.30 pandas==1.3.5 seaborn sklearn matplotlib scikit-fmm tqdm h5py gdown
RUN pip3 install --upgrade pip
RUN apt-get update -y        
RUN apt-get install -y git nvidia-modprobe
RUN pip3 install jupyterlab
# Install required python packages
RUN pip3 install ipywidgets

##### TODO - From the Final Repo Changing this 

# TO COPY the data 
COPY English/ /workspace/

# This Installs All the Dataset
RUN python3 /workspace/python/source_code/dataset.py

## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/jupyter_notebook/
