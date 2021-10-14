# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
wget http://spraakbanken.gu.se/lb/resurser/meningsmangder/webbnyheter2013.xml.bz2 &&
bunzip2 -d webbnyheter2013.xml.bz2 &&
mv webbnyheter2013.xml ./source_code/ &&
wget https://raw.githubusercontent.com/spraakbanken/sb-nltk-tools/master/sb_corpus_reader.py &&
mv sb_corpus_reader.py ./source_code/ &&
cd ./source_code/ &&
python get_nyheterdata.py &&
echo ls &&
mv webnyheter2013.txt ../dataset/SV/ &&
rm -fr webbnyheter2013.xml

