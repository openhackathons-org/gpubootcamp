#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

get_ngc_key_from_environment() {
    # first check the global NGC_API_KEY environment variable
    local ngc_key=$NGC_API_KEY
    # if env variable was not set, and a ~/.ngc/config exists
    # try to get it from there
    if [ -z "$ngc_key" ] && [[ -f "$HOME/.ngc/config" ]]
    then
        ngc_key=$(cat $HOME/.ngc/config | grep apikey | awk '{print $3}')
    fi
    echo $ngc_key
}

docker_pull_and_login_quiet_exit_on_fail() {
    image_exists=$(docker images --filter=reference=$1 -q | wc -l)
    if [[ $image_exists -eq 1 ]]; then
        echo "  > Image $1 exists. Skipping pull."
        return
    fi

    # confirm we're logged in
    # automatically get NGC_API_KEY or request from user if necessary
    NGC_API_KEY="$(get_ngc_key_from_environment)"
    if [ -z "$NGC_API_KEY" ]; then
        read -sp 'Please enter API key for ngc.nvidia.com: ' NGC_API_KEY
        echo
    fi

    # use the API key to run docker login for the NGC registry
    # exit early if the key is invalid, because we won't be able to do anything
    echo "Logging into NGC docker registry if necessary..."
    echo $NGC_API_KEY | docker login -u '$oauthtoken' --password-stdin nvcr.io &> /dev/null
    if [ $? -ne 0 ]; then
        echo 'NGC API Key is invalid. Please check and try again.'
        exit 1
    fi

    echo "  > Pulling $1. This may take some time..."
    docker pull -q $1 &> /dev/null
    if [ $? -ne 0 ]; then
        echo "Error occurred pulling '$1'."
        docker pull $1
        echo "Exiting."
        exit 1
    fi
}

script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
if [ -z "$1" ]; then
    config_path="${script_path}/config.sh"
else
    config_path=$(readlink -f $1)
fi

source $config_path

read -p 'Please enter your JUPYTER PORT NUMBER: ' jupyter_port
read -p 'Please enter your RIVA PORT NUMBER: ' riva_port

echo $riva_port

docker_pull_and_login_quiet_exit_on_fail ${image_client}


cp -r /mnt/shared/jarvis_testing/bootcampnotebooks ./bootcamp

docker run --init -it --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /dev/snd:/dev/snd \
    -v $PWD/jupyter_notebook:/bootcamp \
    -p $jupyter_port:$jupyter_port -p 8000 -p 8001 -p 8002 -p $riva_port \
    --name ${riva_daemon_client} \
    ${image_client}
