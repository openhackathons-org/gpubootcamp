# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t ai-multi-gpu:latest .
# To run: $ sudo docker run --rm -it --gpus=all -p 8888:8888 -p 8000:8000 ai-multi-gpu:latest
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/pytorch:21.03-py3
# Update the repo
RUN apt-get update -y

# Install required python packages
RUN pip3 install tokenizers transformers ipywidgets widgetsnbextension  
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip3 install nvtx ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 htmlmin tldextract sentence-splitter

##### TODO - From the Final Repo Changing this 

# TO COPY the data 
COPY English/ /workspace/

## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/