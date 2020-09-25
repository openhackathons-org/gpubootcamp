# openacc-training-materials
Training materials provided by OpenACC.org.


## Running the Docker container
The code labs have been written using Jupyter notebooks and a Dockerfile has been built to simplify deployment. In order to serve the docker instance for a student, it is necessary to expose port 8000 from the container, for instance, the following command would expose port 8000 inside the container as port 8000 on the lab machine:

    $ docker run --gpus all -rm --it -p 8000:8000 <image>:<tag>

When this command is run, a student can browse to the serving machine on port 8000 using any web browser to access the labs. For instance, from if they are running on the local machine the web browser should be pointed to http://localhost:8000. The `--gpus` flag is used to enable `all` NVIDIA GPUs 
during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. This command may be customized for your
hosting environment.
