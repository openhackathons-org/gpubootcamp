# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t PINN:latest --network=host .
# To run: $ sudo docker run --rm -it --gpus=all --network=host -p 8888:8888 PINN:latest
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/modulus/modulus:22.09

# Install required python packages
RUN pip3 install gdown

# TO COPY the data 
COPY English/ /workspace/

## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/jupyter_notebook/
