# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
#!/usr/bin/env bash
cp create_dir_and_download_pytorch_sif_file.sh ../../../../../../
cd ../../../../../../
mkdir output
mkdir output/sv_gpt3_ckpt/
mkdir dataset

# gdrive_download pre-built **pytorch_21.03.sif** which is needed to run singularity 
# script to download Google Drive files from command line
# not guaranteed to work indefinitely
# taken from Stack Overflow answer:
# http://stackoverflow.com/a/38937732/7002068

gURL=https://drive.google.com/file/d/18-QSZhPhNJS3m9ASTPkjnzsFVgg71MNx/view?usp=sharing
# match more than 26 word characters
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

ggURL='https://drive.google.com/uc?export=download'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
echo -e "Downloading from "$gURL"...\n"
eval $cmd

# gdrive_download toy dataset 

gURL=https://drive.google.com/file/d/17hIXwG6jHgijmBJKq2Z211Hm6AXfQo9C/view?usp=sharing
# match more than 26 word characters
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

ggURL='https://drive.google.com/uc?export=download'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
echo -e "Downloading from "$gURL"...\n"
eval $cmd


### move toy data to correct dirs
mv toydata.zip ./dataset/
cd dataset/
unzip -r toydata.zip
cd ..
ls ./dataset/

### move the Megatron run script to the correct directory
cp ./gpubootcamp/ai/Megatron/English/Python/source_code/Day1-runMegatron-LM_GPT_template.sh ./
echo "done !"