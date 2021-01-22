import gdown
import os

## Challenge dataset

url = 'https://drive.google.com/uc?export=download&id=1_RQlI8FOJRWAZVEYZkpepxAUuhfToEtQ'
output = '/workspace/data/'
gdown.download(url, output, quiet=False,proxy=None)