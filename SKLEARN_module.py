import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

from keras.layers import Dense, Input, Dropout
import keras
from keras import Model
import tensorflow as tf

from sklearn import linear_model

import keras_tuner as kt
from keras.optimizers import Adam, SGD
import statistics as stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.metrics.pairwise import euclidean_distances
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.experimental import enable_halving_search_cv # noqa
# CV
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
    ShuffleSplit,
    HalvingGridSearchCV,                 
)

from scipy.stats import uniform, randint, loguniform

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# KNN
from sklearn.neighbors import KNeighborsRegressor

# multioutput
from sklearn.multioutput import MultiOutputRegressor

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

#ccorr
from cascor.monitor import EarlyStoppingMonitor

import tools_for_classes as tools
import models_list as models_importer

K = 5   # number of folds used in k-fold
VAL_SPLIT = 1/(K-1) #validation split in k-fold
EPOCHS = 500    #max number of epochs
RANDOM_STATE = 42

class SklNet:

    
    def __init__(self, modelName, mode, model, X, y, param_grid, scoring=None, X_test=None, y_test=None):
        
        """Class used to instantiate and train sklearn models.
        ...
        Attributes:
        ----------
        X: data
        y: target
        modelName
        model
        param_grid: obj performing hps search
        mode: classification or regression 
        X_test: test data (used only for monk)
        y_test: test target (used only for monk)"""

        self.X = X
        self.y = y
        self.modelName = modelName
        self.model = model
        self.scoring = scoring #None usually, for knn neg_mean_abs_error
        self.param_grid = param_grid
        self.mode = mode
        self.X_test = X_test
        self.y_test = y_test

#DISPATCHER
    def train(self):
        if self.mode in ['classification', 'clf']:
          best_model, best_params_val, MSE_test, MSE_training, accuracy_training, accuracy_test = self.train_class() 
          return best_model, best_params_val, MSE_test, MSE_training, accuracy_training, accuracy_test 
        
        elif self.mode in ['regressione', 'reg']:
            best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_err= self.train_regression()
            return best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_err

    
    def train_class(self):
        """function for training classification models"""
        self.grid = GridSearchCV(
            self.model,
            param_grid=self.param_grid,
            cv= ShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_STATE), 
            refit = True,
            n_jobs = -1,
            verbose = 0,
            return_train_score=False
        )
        #qui MSE_test non serve a nulla
        best_params_val, MSE_test = self.train_standard(self.X, self.y , self.X_test, self.y_test)

        best_model = self.model.set_params(**best_params_val)
        best_model.fit(self.X, self.y)

        y_pred_training = best_model.predict(self.X)
        MSE_training = mean_squared_error(self.y, y_pred_training)
        accuracy_training = accuracy_score(self.y, y_pred_training)

        y_pred = best_model.predict(self.X_test)
        MSE_test = mean_squared_error(self.y_test, y_pred)

        
        accuracy_test = accuracy_score(self.y_test, y_pred)

        return best_model, best_params_val, MSE_test, MSE_training, accuracy_training, accuracy_test

    #COMMON TRAIN FUN (USED BY BOTH CLASS & REG)
    #perform a grid search over a set of hyperp and return the best combination (with best val error)
    def train_standard(self, X_train, y_train, X_test, y_test):


        self.grid.fit(X_train, y_train)

        best_params_val = self.grid.best_params_ 


        chosen_model = self.grid.best_estimator_

 
        grid_predictions_test = chosen_model.predict(X_test)
        
        if(self.mode in ['clf','classification']):
            test_error = mean_squared_error(y_test, grid_predictions_test)
        else:
            test_error = tools.MEE_metric(y_test, grid_predictions_test)
    

        print("-------------------") 
        return best_params_val, test_error
    
    #SPECIFIC TRAIN FUN, USED ONLY BY REG (PERFORMS K-FOLD)
    def train_regression(self):
        outer_kfold = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)

        self.grid = GridSearchCV(
            self.model,
            param_grid=self.param_grid,
            cv= ShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_STATE), 
            refit = True,
            n_jobs = -1,
            verbose = 0,
            return_train_score=False,
            scoring = self.scoring
        )
    
    
        val_MEE_list=[]      
        best_hps_list = []
        test_error_list = []

        
        print("############################")
        print("START OF OUTER K-FOLD")
        for dev_idx, test_idx in outer_kfold.split(self.X):        
            X_dev, X_test = self.X[dev_idx], self.X[test_idx]
            y_dev, y_test = self.y[dev_idx], self.y[test_idx]

            best_params_val, MEE_test = self.train_standard(X_dev, y_dev, X_test, y_test)
            best_hps_list.append(best_params_val)
            test_error_list.append(MEE_test)
            
            
        
        print("END OF OUTER K-FOLD")
        print("############################")
        print("############################")

        #RE-DEFINE THE MODEL WITH EMPTY HYPERP
        #select the best model (with best val error)
        print("############################")
        print("START FINAL MODEL SELECTION (done with hold-out)")
        if(self.modelName in ['KNN','knn']):
            self.model = models_importer.build_KNN_Pipe_Reg()

        if(self.modelName in ['SVR','svr']):
            self.model = models_importer.build_SVR(EPOCHS) 

        if(self.modelName in ['RIDGE','ridge']):
            self.model = models_importer.build_RidgeRegressor(EPOCHS) 
        
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=VAL_SPLIT, random_state=RANDOM_STATE)

        for i in range(K):
            self.model.set_params(**best_hps_list[i])
            history = self.model.fit(X_train, y_train)

            #evaluation on training set just for print
            y_pred_tr = self.model.predict(X_train)
            MEE_train = tools.MEE_metric(y_train, y_pred_tr)


            y_pred = self.model.predict(X_val)

            MEE_val = tools.MEE_metric(y_val, y_pred)

            #check the val_loss and save the best model so far
            val_MEE_list.append(MEE_val)
            if (MEE_val == min(val_MEE_list)):
                best_training_err = MEE_train
                best_model_MEE_val = MEE_val 
                best_params = best_hps_list[i] 
                best_model = self.model 
            print("------------------")    

        print("END FINAL MODEL SELECTION")
        print("############################")
        mean_test_error = round(stats.mean(test_error_list),2)
        stdev_test_error = round(stats.stdev(test_error_list),2)
    
        return best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_err
