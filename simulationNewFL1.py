# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:37:13 2023

@author: vikas
"""

import os
import time
from multiprocessing import Process
from typing import Tuple
import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

'''
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
'''

import time
import shutil
import itertools

# import data handling tools 
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')



from flwr.server.strategy import FedAvg as FA
import dataset as dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"


DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

from keras.callbacks import CSVLogger

path = 'Results\\FL\\IID\\'



from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plot_multiclass_roc(clf, X_test, y_test, n_classes,fcntr):
    """
    Plot multi-class ROC curves with micro and macro averages
    """
    # Binarize the output for multiclass
    y_test_bin = label_binarize(y_test, classes=range(0, n_classes))

    y_score = clf.predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(4, 4))
    colors = cycle(['blue', 'red', 'green','Yellow','black', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC - IID Client - '+fcntr)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("IID-ROC-Client"+fcntr+".jpeg")


def create_model():
    # Create Model Structure
    model = keras.Sequential()
    model.add(Dense(128 , input_shape=(5,1) , activation="relu" , name="Hidden_Layer_1"))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'Precision', 'Recall'])
    
    model.summary()
    
    return model




def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    print("number clients ", num_clients )

    
    model = create_model()
    model.summary()
    weights = model.get_weights()

    # Serialize ndarrays to `Parameters`
    parameters = fl.common.ndarrays_to_parameters(weights)
    strategy = FA(min_available_clients=num_clients, min_fit_clients=num_clients, fraction_fit=fraction_fit, initial_parameters=parameters)
    # Exposes the server by default on port 8080
    fl.server.start_server(server_address = "127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy)


def start_client(dataset: DATASET, fcntr) -> None:
    """Start a single client with the provided dataset."""
    
    
    model = create_model()
    
    csv_logger = CSVLogger(path+str(fcntr)+"IR.csv", append=True)
    csv_logger1 = CSVLogger(path+str(fcntr)+"_Eval_IR.csv", append=True)
    (x_train, y_train), (x_test, y_test) = dataset


    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            """Return current weights."""
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=10,callbacks=[csv_logger])
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            """Evaluate using provided parameters."""
            model.set_weights(parameters)
            loss, accuracy, recall,Precision = model.evaluate(x_test, y_test,callbacks=[csv_logger1])
            pred = model.predict(x_test)
            pred = np.argmax(pred, axis=1)
            
            label = np.argmax(y_test, axis=1)
            fl = open(path+"EvalRes"+str(fcntr)+".csv", "a+")
            fl.write(str(fcntr)+","+str(loss)+","+str(accuracy)+","+str(recall)+","+str(Precision)+"\n")
            fl.close()
            dfNor3 = pd.DataFrame(confusion_matrix(label, pred))
            
            dfNor3.to_csv(path+"Conf//Confusion_IID"+str(fcntr)+".csv",mode='a')
            
            auc_scores = plot_multiclass_roc(model, x_test, y_test, y_test.shape[1],str(fcntr))

            
            return loss, len(x_test), {"accuracy": accuracy}
    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())

def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    processes = []
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    time.sleep(2)
    
    partitions = dataset.load(num_partitions=num_clients)
    fcntr = 0
    
    for partition in partitions:
        fcntr = fcntr+1
        client_process = Process(target=start_client, args=(partition, fcntr))
        client_process.start()
        processes.append(client_process)

    for p in processes:
        print("vikas",p)
        p.join()


if __name__ == "__main__":
    run_simulation(num_rounds=10, num_clients=4, fraction_fit= 1)