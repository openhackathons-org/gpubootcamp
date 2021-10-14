# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
import json
import os, sys
import numpy as np
import nltk
from sb_corpus_reader import SBCorpusReader
import random

def write2csv(out_path, fname, sents):
    f=open(out_path+fname,'a')
    for s in sents:
        if len(s)>=2:
            s_text=' '.join(s)
            f.write(s_text+'\n')
    print("finish processing ",fname)
    f.close()
    
out_path='./'
xml_f=out_path+'webbnyheter2013.xml'
if xml_f.endswith('.xml') :    
    corpus = SBCorpusReader(xml_f)
    sents=corpus.sents()
    print(sents[:2])
    #n=len(sents)
    #rn=random.randint(0,n-1)
    #print("a random sample of sentence : \n".format(' '.join(sents[rn])))
    fname='webnyheter2013.txt'  
    print("write to : ",fname)
    write2csv(out_path,fname,sents)
    print('-----'*10)