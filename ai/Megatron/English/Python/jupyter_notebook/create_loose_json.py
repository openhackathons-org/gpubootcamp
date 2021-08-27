# -*- coding: utf-8 -*-
import json
import os, sys
import numpy as np
import argparse
import numpy as np
import torch
import os
import pandas as pd
import time
#also filter duplicates
def main(args):
    with open(args.infile,'r', encoding='utf-8',errors='ignore') as fin:
        with open(args.outfile,'a', encoding='utf-8') as fout:
            lines=fin.readlines()
            i=0
            for line in lines: 
                if line.strip() not in ['\n','\t','',' ','\r\n'] : # make sure it's not empty
                    d={'text':line.strip()} 
                    #print(d.items())
                    data = json.dumps(d,ensure_ascii=False)
                    fout.write(data)
                    fout.write('\n')
                    i+=1
                    if i%1000000==0:
                        print("process {} documents so far ...".format(str(i)))
                        print("example: ", line)            
    fin.close()
    fout.close()
    print("finished processing {} lines to loose json format".format(str(i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str,  help='input file path')
    parser.add_argument('--outfile', default=None, type=str, 
                        help='output file path')
    args = parser.parse_args()
    main(args)