import math 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import tikzplotlib

'''
functions computing side tasks
'''

#evaluation of the Mean Euclidean Error
def MEE_metric(y_test, y_pred):
    distance = 0
    for i in range (len(y_pred)):
        diff = y_pred[i] - y_test[i]
        distance += np.linalg.norm(diff)
    MEE = distance/len(y_pred)
    return MEE


#evaluation of the size of the hyperparameters search space 
def get_search_spaze_size(hyperp):
    size = 1
    for key in hyperp:
        if(key in ['activation_hidden','activation_output', 'output_units', 'metric']):
            pass
        else:
            size*= len(hyperp[key])
    return size

#dataset preprocessing
def preprocessing(path):
    #path='./ML-CUP_prepro.csv'
    df = pd.read_csv(path)
    X_df = df.drop(['Target_1', 'Target_2'], axis=1)
    y_df = df[["Target_1", "Target_2"]]
    X = X_df.values
    y = y_df.values
    return X,y

#plot function
def plot(history, mode, name):
    """Plot loss and accuracy
    .....
    Parameters
    ----------
    history: keras History obj
        model.fit() return
   """
    print(history.history.keys())
    #Train and validation accuracy 
    plt.figure()
    if mode in ['classification', 'clf']:
        plt.figure(figsize=(9, 6))
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss' )
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
        plt.xlabel('Epochs')
        plt.ylabel('Loss(MSE)')
        plt.legend()
        plt.title('TR and VS Loss')
        plt.subplot(2, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy',  linestyle='dashed')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('TR and VS Accuracy')
        #plt.suptitle(name)
        folder = 'Monk-plots'
        save_plot(folder, name)
        plt.show(block=False)

    
    elif mode in ['regression', 'reg']:
        plt.figure(figsize=(9, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss',  linestyle='dashed')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss(MSE)')
        plt.title('TR and VS Loss')
        plt.legend()
        #plt.suptitle(name)
        folder = 'ML-CUP-plots'
        save_plot(folder, name)
        plt.show(block=False)


def save_plot(folder, name):
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig(os.path.join(folder, name + f'_plot.pdf'), bbox_inches='tight', dpi=600)
    folder = folder + '/tikz/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    tikzplotlib.save(os.path.join(folder, name + '_plot.tex'))

def get_param_cc(n_in, n_out, n_hid):
    tot=n_in*n_out + n_hid*(n_in + n_out) + n_hid*(n_hid -1)/2
    return tot

        
    
if __name__=='__main__':
    #y_test=np.ndarray([[1], [1], [0]])
    #y_pred=np.ndarray([[0], [1], [0]])
    y_test=np.array([[1,0], [0,0],[1,1]])
    y_pred=np.array([[0,1], [1,0],[1,1]])
    print(MEE_metric(y_pred, y_test))
   
