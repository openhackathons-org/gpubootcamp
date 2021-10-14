# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
#!/bin/bash
export PATH="${PATH}:${HOME}/.local/bin"
input="NVdevblog_urls.txt"
CNT=0

while IFS= read -r line
do
  c="./htmls/response_${CNT}.html"
  echo "$c"
  scrapy fetch --nolog $line > $c  
  ((CNT+=1))
done < "$input"