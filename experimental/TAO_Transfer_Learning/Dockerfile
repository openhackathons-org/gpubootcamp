# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

## To build the docker container, run: $ sudo docker build -t ai-tao:1.0 .
## To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 -p 8000:8000 ai-tao:1.0
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3
# Update the repo
RUN apt-get update -y
# Install required dependencies
RUN apt-get install -y git nvidia-modprobe
#RUN pip3 install --no-cache-dir jupyter
RUN pip3 install jupyterlab
# Install required python packages
RUN pip3 install ipywidgets
RUN pip3 install gdown



# TO COPY the data 
COPY English/ /workspace/tlt-experiments


# To Download VOC dataaset
#RUN mkdir /workspace/tlt-experiments/data
#RUN curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -o ~/English/tao_experiments/data/VOCtrainval_11-May-2012.tar

RUN python3 /workspace/tlt-experiments/source_code/dataset.py



## Uncomment this line to run Jupyter notebook by default
#CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/tlt-experiments
