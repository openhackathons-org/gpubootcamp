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
        ngc_key=$(cat $HOME/.ngc/config | grep ^apikey | awk '{print $3}')
    fi
    echo $ngc_key
}

docker_pull() {
    image_exists=$(docker images --filter=reference=$1 -q | wc -l)
    if [[ $image_exists -eq 1 ]]; then
        echo "  > Image $1 exists. Skipping."
        return
    fi
    attempts=3
    echo "  > Pulling $1. This may take some time..."
    for ((i = 1 ; i <= $attempts ; i++)); do
        docker pull -q $1 &> /dev/null
        if [ $? -ne 0 ]; then
            echo "  > Attempt $i out of $attempts failed"
            if [ $i -eq $attempts ]; then
                echo "Error occurred pulling '$1'."
                docker pull $1
                echo "Exiting."
                exit 1
            else
                echo "  > Trying again..."
                continue
            fi
        else
            break
        fi
    done
}

check_docker_version() {
    version_string=$(docker version --format '{{.Server.Version}}')
    if [ $? -ne 0 ]; then
        echo "Unable to run Docker. Please check that Docker is installed and functioning."
        exit 1
    fi
    maj_ver=$(echo $version_string | awk -F. '{print $1}')
    min_ver=$(echo $version_string | awk -F. '{print $2}')
    if [ "$maj_ver" -lt "19" ] || ([ "$maj_ver" -eq "19" ] && [ "$min_ver" -lt "03" ]); then
        echo "Docker version insufficient. Please use Docker 19.03 or later"
        exit 1;
    fi
}

# BEGIN SCRIPT
check_docker_version

# load config file
script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
if [ -z "$1" ]; then
    config_path="${script_path}/config.sh"
else
    config_path=$(readlink -f $1)
fi

if [[ ! -f $config_path ]]; then
    echo 'Unable to load configuration file. Override path to file with -c argument.'
    exit 1
fi
source $config_path || exit 1

read -p 'Please enter your RIVA SERVER PORT NUMBER: ' riva_port

sed -i 's/riva_speech_api_port="50051"/riva_speech_api_port="'$riva_port'"/' $config_path

servername="riva_speech_$riva_port"
clientname="riva_client_$riva_port"

echo $riva_port
echo $servername
echo $clientname

sed -i 's/riva_daemon_speech="riva-speech"/riva_daemon_speech="'$servername'"/' $config_path
sed -i 's/riva_daemon_client="riva-client"/riva_daemon_client="'$clientname'"/' $config_path

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

# pull all the requisite images we're going to need
echo "Pulling required docker images if necessary..."
echo "Note: This may take some time, depending on the speed of your Internet connection."
# pull the speech server if any of asr/nlp/tts services are requested
if [ "$service_enabled_asr" = true ] || [ "$service_enabled_nlp" = true ] || [ "$service_enabled_tts" = true ]; then
    echo "> Pulling Riva Speech Server images."
    docker_pull $image_speech_api
    docker_pull $image_client
    docker_pull $image_init_speech
fi


if [ "$use_existing_rmirs" = false ]; then
    echo
    echo "Downloading models (RMIRs) from NGC..."
    echo "Note: this may take some time, depending on the speed of your Internet connection."
    echo "To skip this process and use existing RMIRs set the location and corresponding flag in config.sh."

    # build up commands to download from NGC
    if [ "$service_enabled_asr" = true ] || [ "$service_enabled_nlp" = true ] || [ "$service_enabled_tts" = true ]; then
        gmr_speech_models=""
        if [ "$service_enabled_asr" = true ]; then
            for model in ${models_asr[@]}; do
                gmr_speech_models+=" $model"
            done
        fi
        if [ "$service_enabled_nlp" = true ]; then
            for model in ${models_nlp[@]}; do
                gmr_speech_models+=" $model"
            done
        fi
        if [ "$service_enabled_tts" = true ]; then
            for model in ${models_tts[@]}; do
                gmr_speech_models+=" $model"
            done
        fi

        # download required images
        docker run --init -it --rm --gpus '"'$gpus_to_use'"'  \
          -v $riva_model_loc:/data \
          -e "NGC_CLI_API_KEY=$NGC_API_KEY" \
          -e "NGC_CLI_ORG=nvidia" \
          --name riva-service-maker \
          $image_init_speech download_ngc_models $gmr_speech_models

        if [ $? -ne 0 ]; then
            echo "Error in downloading models."
            exit 1
        fi
    fi
fi

# convert all rmirs
echo
echo "Converting RMIRs at $riva_model_loc/rmir to Riva Model repository."

set -x
docker run --init -it --rm --gpus '"'$gpus_to_use'"' \
  -v $riva_model_loc:/data \
  -e "MODEL_DEPLOY_KEY=${MODEL_DEPLOY_KEY}" \
          --name riva-service-maker \
  $image_init_speech deploy_all_models /data/rmir /data/models

echo
echo "Riva initialization complete. Run ./riva_start.sh to launch services."
