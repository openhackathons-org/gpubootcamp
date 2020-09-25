
# To run this dockerfile you need to present port 8888 and provide a hostname. 
# For instance:
#   $ nvidia-docker run --rm -it -p "8888:8888" -e HOSTNAME=foo.example.com openacc-labs:latest
FROM nvcr.io/hpc/pgi-compilers:ce

RUN apt update && \
    apt install -y --no-install-recommends python3-pip 

ADD appName/ /labs
WORKDIR /labs
