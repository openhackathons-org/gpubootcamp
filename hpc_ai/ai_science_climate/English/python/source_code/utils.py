# Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import tensorflow as tf
import numpy as np
def dummy():
    pass
def load_image(name,interpolation = cv2.INTER_AREA):
    img=cv2.imread(name,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inter_area = cv2.resize(img,(256,256),interpolation=interpolation)
    start_pt= np.random.randint(24,size=2)
    end_pt = start_pt + [232,232]
    img = inter_area[start_pt[0]:end_pt[0],start_pt[1]:end_pt[1]]
    return img

def load_dataset(augment_fn = dummy):
    import os
    import cv2
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from scipy import interpolate 
    import matplotlib.pyplot as plt

    #Variables to be used later
    filenames = []
    labels =[]
    i = 0  
    #Read CSV file using Pandas
    df = pd.read_csv('atlantic_storms.csv')

    dir ='Dataset/tcdat/'
    a = os.listdir(dir)

    file_path = "Dataset/Aug/"
    directory = os.path.dirname(file_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)   
    aug = 0 
    for j in a :
        c = os.listdir(dir+'/'+j)
        for k in c :
            d = os.listdir(dir+'/'+j+'/'+k)
            for l in d :
                print('.',end='')
                start_year= '20'+j[2:]+ '-01-01'
                end_year= '20'+j[2:]+ '-12-31'
                cyc_name = l[4:]
                mask = (df['date'] > start_year ) & (df['date'] <= end_year ) & ( df['name'] == cyc_name )
                cyc_pd = df.loc[mask]
                first = (datetime.strptime(cyc_pd['date'].iloc[0], "%Y-%m-%d %H:%M:%S"))
                last = (datetime.strptime(cyc_pd['date'].iloc[-1], "%Y-%m-%d %H:%M:%S"))
                text_time=[]
                text_vel=[]
                for q in range(len(cyc_pd['date'])):
                    text_vel.append(cyc_pd['maximum_sustained_wind_knots'].iloc[q])
                    text_time.append((datetime.strptime(cyc_pd['date'].iloc[q],"%Y-%m-%d %H:%M:%S")-first).total_seconds())
                func = interpolate.splrep(text_time,text_vel)
                e = os.listdir(dir+'/'+j+'/'+k+'/'+l+'/ir/geo/1km')
                e.sort()
                for m in e :
                    try :
                        time=(datetime.strptime(m[:13], "%Y%m%d.%H%M"))
                        name = dir+j+'/'+k+'/'+l+'/ir/geo/1km/'+m
                        if(time>first and time < last):
                            val = int(interpolate.splev((time-first).total_seconds(),func))
                            filenames.append(name)
                            if val <=20 :
                                labels.append(0)
                            elif val>20 and val <=33 :
                                labels.append(1)
                            elif val>33 and val <=63 :
                                labels.append(2)
                            elif val>63 and val <=82 :
                                labels.append(3)
                            elif val>82 and val <=95 :
                                labels.append(4)
                            elif val>95 and val <=112 :
                                labels.append(5)
                            elif val>112 and val <=136 :
                                labels.append(6)
                            elif val>136 :
                                labels.append(7)
                            i = augment_fn(name,labels[-1],filenames,labels,i)
                    except :
                        pass
    print('')
    print(len(filenames)) 
     # Shuffle The Data
    import random
    # Zip Images with Appropriate Labels before Shuffling
    c = list(zip(filenames, labels))
    random.shuffle(c)
    #Unzip the Data Post Shuffling
    filenames, labels = zip(*c)
    filenames = list(filenames)
    labels = list(labels)
    return filenames,labels
# Let's make a Validation Set with 10% of the Original Data with 1.25% contribution of every class
def make_test_set(filenames,labels,val=0.1):
    classes = 8
    j=0
    val_filenames=[]
    val_labels=[]
    new = [int(val*len(filenames)/classes)]*classes
    print(new)
    try:
        for i in range(len(filenames)):
            if(new[labels[i]]>0):
                val_filenames.append(filenames[i])
                val_labels.append(labels[i])
                new[labels[i]] = new[labels[i]]-1
                del filenames[i]
                del labels[i]
    except :
        pass
    
     # Shuffle The Data
    import random
    # Zip Images with Appropriate Labels before Shuffling
    c = list(zip(val_filenames, val_labels))
    random.shuffle(c)
    #Unzip the Data Post Shuffling
    val_filenames, val_labels = zip(*c)
    val_filenames = list(val_filenames)
    val_labels = list(val_labels)
    from collections import Counter
    print(Counter(labels))
    return val_filenames,val_labels  

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    #Resize Image 
    image = tf.image.resize(image, [232, 232])
    
    return image, label

def make_dataset(train_in,test_in,val_in):
    import tensorflow as tf
    train = tf.data.Dataset.from_tensor_slices((train_in[0], train_in[1]))
    train = train.shuffle(len(train_in[0]))
    train = train.map(parse_function,num_parallel_calls=8)
    train = train.batch(train_in[2])
    train = train.prefetch(1)
    test = tf.data.Dataset.from_tensor_slices((test_in[0], test_in[1]))
    test = test.shuffle(len(test_in[0]))
    test = test.map(parse_function, num_parallel_calls=8)
    test = test.batch(test_in[2])
    test = test.prefetch(1)
    val = tf.data.Dataset.from_tensor_slices((val_in[0],val_in[1] ))
    val = val.map(parse_function, num_parallel_calls=8)
    val = val.batch(val_in[2])
    val = val.prefetch(1)
    return train,test,val

