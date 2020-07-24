
# To run this dockerfile you need to present port 8888 and provide a hostname. 
# For instance:
#   $ nvidia-docker run --rm -it -p "8888:8888" -e HOSTNAME=foo.example.com openacc-labs:latest
FROM nvcr.io/hpc/pgi-compilers:ce

RUN apt update && \
    apt install -y --no-install-recommends python3-pip python3-setuptools nginx zip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir jupyter
ADD docker-configs/default /etc/nginx/sites-available/default

ADD labs/ /labs
WORKDIR /labs
CMD service nginx start && jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs