# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Enable or Disable Riva Services
service_enabled_asr=true
service_enabled_nlp=true
service_enabled_tts=true

# Specify one or more GPUs to use
# specifying more than one GPU is currently an experimental feature, and may result in undefined behaviours.
gpus_to_use="device=0"

# Specify the encryption key to use to deploy models
MODEL_DEPLOY_KEY="tlt_encode"

# Locations to use for storing models artifacts
#
# If an absolute path is specified, the data will be written to that location
# Otherwise, a docker volume will be used (default).
#
# riva_init.sh will create a `rmir` and `models` directory in the volume or
# path specified. 
#
# RMIR ($riva_model_loc/rmir)
# Riva uses an intermediate representation (RMIR) for models
# that are ready to deploy but not yet fully optimized for deployment. Pretrained
# versions can be obtained from NGC (by specifying NGC models below) and will be
# downloaded to $riva_model_loc/rmir by `riva_init.sh`
# 
# Custom models produced by NeMo or TLT and prepared using riva-build
# may also be copied manually to this location $(riva_model_loc/rmir).
#
# Models ($riva_model_loc/models)
# During the riva_init process, the RMIR files in $riva_model_loc/rmir
# are inspected and optimized for deployment. The optimized versions are
# stored in $riva_model_loc/models. The riva server exclusively uses these
# optimized versions.
riva_model_loc="riva-model-repo"

# The default RMIRs are downloaded from NGC by default in the above $riva_rmir_loc directory
# If you'd like to skip the download from NGC and use the existing RMIRs in the $riva_rmir_loc
# then set the below $use_existing_rmirs flag to true. You can also deploy your set of custom
# RMIRs by keeping them in the riva_rmir_loc dir and use this quickstart script with the
# below flag to deploy them all together.
use_existing_rmirs=false

# Ports to expose for Riva services
riva_speech_api_port="50051"
riva_vision_api_port="60051"

# NGC orgs
riva_ngc_org="nvidia"
riva_ngc_team="riva"
riva_ngc_image_version="1.4.0-beta"
riva_ngc_model_version="1.4.0-beta"

# Pre-built models listed below will be downloaded from NGC. If models already exist in $riva-rmir
# then models can be commented out to skip download from NGC

########## ASR MODELS ##########

models_asr=(
### Punctuation model
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_base:${riva_ngc_model_version}"

### Citrinet-1024 Streaming w/ CPU decoder, best latency configuration
    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_citrinet_1024_asrset1p7_streaming:${riva_ngc_model_version}"

### Citrinet-1024 Streaming w/ CPU decoder, best throughput configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_citrinet_1024_asrset1p7_streaming_throughput:${riva_ngc_model_version}"

### Citrinet-1024 Offline w/ CPU decoder, 
    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_citrinet_1024_asrset1p7_offline:${riva_ngc_model_version}"

### Jasper Streaming w/ CPU decoder, best latency configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_jasper_english_streaming:${riva_ngc_model_version}"

### Jasper Streaming w/ CPU decoder, best throughput configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_jasper_english_streaming_throughput:${riva_ngc_model_version}"

###  Jasper Offline w/ CPU decoder
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_jasper_english_offline:${riva_ngc_model_version}"
 
### Quarztnet Streaming w/ CPU decoder, best latency configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_quartznet_english_streaming:${riva_ngc_model_version}"

### Quarztnet Streaming w/ CPU decoder, best throughput configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_quartznet_english_streaming_throughput:${riva_ngc_model_version}"

### Quarztnet Offline w/ CPU decoder
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_quartznet_english_offline:${riva_ngc_model_version}"

### Jasper Streaming w/ GPU decoder, best latency configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_jasper_english_streaming_gpu_decoder:${riva_ngc_model_version}"

### Jasper Streaming w/ GPU decoder, best throughput configuration
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_jasper_english_streaming_throughput_gpu_decoder:${riva_ngc_model_version}"

### Jasper Offline w/ GPU decoder
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_jasper_english_offline_gpu_decoder:${riva_ngc_model_version}"
)

########## NLP MODELS ##########

models_nlp=(
### Bert base Punctuation model
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_base:${riva_ngc_model_version}"

### BERT base Named Entity Recognition model fine-tuned on GMB dataset with class labels LOC, PER, ORG etc.
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_named_entity_recognition_bert_base:${riva_ngc_model_version}"

### BERT Base Intent Slot model fine-tuned on weather dataset.
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_intent_slot_bert_base:${riva_ngc_model_version}"

### BERT Base Question Answering model fine-tuned on Squad v2.
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_question_answering_bert_base:${riva_ngc_model_version}"

### Megatron345M Question Answering model fine-tuned on Squad v2.
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_question_answering_megatron:${riva_ngc_model_version}"

### Bert base Text Classification model fine-tuned on 4class (weather, meteorology, personality, nomatch) domain model.
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_text_classification_bert_base:${riva_ngc_model_version}"
)

########## TTS MODELS ##########

models_tts=(
   "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_ljspeech:${riva_ngc_model_version}"
#   "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_tacotron_waveglow_ljspeech:${riva_ngc_model_version}"
)

NGC_TARGET=${riva_ngc_org}
if [[ ! -z ${riva_ngc_team} ]]; then
  NGC_TARGET="${NGC_TARGET}/${riva_ngc_team}"
else
  team="\"\""
fi

# define docker images required to run Riva
image_client="nvcr.io/${NGC_TARGET}/riva-speech-client:${riva_ngc_image_version}"
image_speech_api="nvcr.io/${NGC_TARGET}/riva-speech:${riva_ngc_image_version}-server"

# define docker images required to setup Riva
image_init_speech="nvcr.io/${NGC_TARGET}/riva-speech:${riva_ngc_image_version}-servicemaker"

# daemon names
riva_daemon_speech="riva-speech"
riva_daemon_client="riva-client"
