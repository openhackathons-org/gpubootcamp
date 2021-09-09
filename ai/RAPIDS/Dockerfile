# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# Select Base Image 
#FROM rapidsai/rapidsai:cuda10.1-runtime-ubuntu18.04-py3.7
FROM rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.7

# Update the repo
RUN apt-get update -y
# Install required dependencies
RUN apt-get install -y libsm6 libxext6 libxrender-dev git 

# Install required python packages
RUN pip install gdown

# TO COPY the data 
COPY English/Python /workspace

# Copy the Python file for downloading dataset 
WORKDIR /workspace/data
RUN python3 /workspace/source_code/dataset.py

WORKDIR /workspace
CMD jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace
