# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t ai-multi-gpu:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 -p 8000:8000 ai-multi-gpu:latest
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
#FROM nvcr.io/nvidia/tensorflow:21.07-tf2-py3 
FROM nvcr.io/nvidia/tensorflow:22.03-tf2-py3

# Update the repo
RUN apt-get update -y
# Install required dependencies
RUN apt-get install -y git nvidia-modprobe
# Install required python packages
RUN pip3 install ipywidgets

##### TODO - From the Final Repo Changing this 

# TO COPY the data 
COPY English/ /workspace/

# To Download Wikitext-2 dataaset
RUN mkdir /workspace/python/source_code/Data
RUN curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -o /workspace/python/source_code/Data/wikitext-2-v1.zip
RUN unzip /workspace/python/source_code/Data/wikitext-2-v1.zip -d /workspace/python/source_code/Data

## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/
