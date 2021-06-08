# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $ sudo docker build -t ai-science-cfd:latest --network=host .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 ai-science-cfd:latest
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3
# Update the repo
RUN apt-get update
# Install required dependencies
RUN apt-get install -y libsm6 libxext6 libxrender-dev git
# Install required python packages
RUN pip3 install opencv-python==4.1.2.30 pandas seaborn sklearn matplotlib scikit-fmm tqdm h5py gdown

# TO COPY the data
COPY English/ /workspace/
#COPY English/python/jupyter_notebook/CFD /workspace/CFD/
#COPY English/python/jupyter_notebook/Intro_to_DL /workspace/Intro_to_DL/
#COPY English/Start_Here.ipynb /workspace/

# Make a directory for Data
RUN mkdir /workspace/python/jupyter_notebook/CFD/data

# Copy the Python file for downloading dataset 
#COPY English/python/source_code/dataset.py /workspace/
# This Installs All the Dataset
#RUN python3 /workspace/dataset.py
RUN python3 /workspace/python/source_code/dataset.py

## Uncomment this line to run Jupyter notebook by default
#CMD jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
CMD jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/jupyter_notebook/

