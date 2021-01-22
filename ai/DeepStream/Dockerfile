# Select Base Image
FROM nvcr.io/nvidia/deepstream:5.0-20.07-triton
# Update the repo
RUN apt-get update
# Install required dependencies
RUN apt-get install ffmpeg python3-gi -y
# Install required python packages
WORKDIR /opt/nvidia/deepstream/deepstream/lib
RUN python3 setup.py install
WORKDIR /opt/nvidia/deepstream/deepstream-5.0
RUN pip3 install jupyterlab
COPY English /opt/nvidia/deepstream/deepstream-5.0

CMD jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8889 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-5.0/python
