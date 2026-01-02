import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus


algos = ['ResNet152V2','InceptionResNetV2','MobileNet','DenseNet201', 'EfficientNetB3']

prep = ['resnet_v2','inception_resnet_v2','mobilenet','densenet', 'efficientnet']



from keras.models import Model
import numpy as np
import cv2
import numpy as np
from sklearn import preprocessing
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from importlib import reload

import glob
files = glob.glob('FData/*/*.*')
files
len(files)

#for i in range(0,len(algos)):
for i in range(0,1):
    algorithm = 'from tensorflow.keras.applications.'+prep[i]+ ' import '+algos[i]
    preprocess = 'from tensorflow.keras.applications.'+prep[i]+ ' import preprocess_input'
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    model.summary()

    shape = str(model.inputs[0]).split('shape=')[1].split(')')[0][7:]
    
    value=[]
    label=[]
    for f in files:
        input_img = cv2.imread(f)
        exec('input_img = np.resize(input_img, ('+shape+'))')
        input_img.shape
        exec('input_img = input_img.reshape(1,'+shape+')') 
        input_img=preprocess_input(input_img)
        value.append(model.predict(input_img))
        label.append(f.split('\\')[-2])
        print(f)
        
    values = np.array(value)
    values.shape

    le = preprocessing.LabelEncoder()
    le.fit(label)
    labels = le.transform(label)
    labels

    values_dec = values.reshape(values.shape[0], values.shape[1]*values.shape[2])