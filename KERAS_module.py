import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras_tuner as kt
import statistics as stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



import numpy as np
import matplotlib.pyplot as plt


# CV
from sklearn.model_selection import (
    train_test_split,
    KFold,                
)


# metrics
from sklearn.metrics import mean_squared_error, accuracy_score

#ccorr
from cascor.monitor import EarlyStoppingMonitor

import tools_for_classes as tools



K = 5   # number of folds used in k-fold
VAL_SPLIT = 1/(K-1) #validation split in k-fold
EPOCHS = 500    #max number of epochs
RANDOM_STATE = 42

class KerasNet:   

    
    def __init__(self, modelName, mode, X, y, tunerParameters, modelBuilder, X_test = None, y_test=None):

        """Class used to instantiate and train keras models.
        ...
        Attributes:
        ----------
        X: data
        y: target
        tunerParameters: parameters for hps space search (batch size, max_trials)
        modelName
        mode: classification or regression
        modelBuilder: funtion returning the hypermodel 
        X_test: test data (used only for monk)
        y_test: test target (used only for monk)"""

        self.X = X
        self.y = y
        self.tunerParameters = tunerParameters
        self.modelName = modelName
        self.mode = mode
        self.modelBuilder = modelBuilder
        self.X_test = X_test
        self.y_test = y_test

        if mode in ['regression', 'reg']:
            self.metric = 'val_loss'
            self.callbacks = [EarlyStopping(monitor='val_loss', patience=20, min_delta = 3e-1, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, min_delta = 1e-2, verbose=0)]  
                
        if mode in ['classification', 'clf']:
            self.metric = 'accuracy'
            self.callbacks = [EarlyStopping(monitor='val_loss', patience=20, min_delta = 1e-2, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, min_delta = 1e-3, verbose=0)]  

        
        
        
    
    
    #DISPATCHER
    #invoke the right train function accordingly to the task needed
    def train(self):
        if self.mode in ['classification', 'clf']:
            
            if self.modelName in ['CC', 'cc']:    
                ccnn, MSE_training, accuracy_training, best_model_MSE_TEST, accuracy_test, ccnn.num_hidden = self.train_CC_clf(self.tunerParameters['batch_size'])
                return ccnn, MSE_training, accuracy_training, best_model_MSE_TEST, accuracy_test, ccnn.num_hidden
            else:
                best_model, best_hps, MSE_training, accuracy_training, best_model_MSE_TEST, accuracy_test  = self.train_class()
                return best_model, best_hps, MSE_training, accuracy_training, best_model_MSE_TEST, accuracy_test

        if self.modelName in ['CC', 'cc']:
            best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_error = self.train_CC_reg(self.tunerParameters['batch_size'])
            return best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_error
        else:
            best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_error = self.train_regression()
            return best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_error


    #trainin fun for classification task
    def train_class(self):
        """function for training classification models"""

        best_hps, _, tuner= self.train_standard(self.X, self.y, self.X_test, self.y_test)
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(self.X, self.y, validation_split=VAL_SPLIT,epochs=EPOCHS, 
                    batch_size=self.tunerParameters['batch_size'], verbose=0, callbacks = self.callbacks
                    )

        y_pred_training = best_model.predict(self.X)
        MSE_training = mean_squared_error(self.y, y_pred_training)
        y_pred_training = np.rint(y_pred_training)
        accuracy_training = accuracy_score(self.y, y_pred_training)


        y_pred = best_model.predict(self.X_test)
        best_model_MSE_TEST = mean_squared_error(self.y_test, y_pred)

        m = tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy", dtype=None, threshold=0.5
            )
        m.update_state(self.y_test, y_pred)
        accuracy_test=m.result().numpy()
        
        tools.plot(history=history, mode=self.mode, name=self.modelName)


        return best_model, best_hps, MSE_training, accuracy_training, best_model_MSE_TEST, accuracy_test


    #CASCADE CORRELATION for reg
    def train_CC_reg(self, batch_size):
        outer_kfold = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)

        val_MEE_list=[]      
        best_models_list = []
        test_error_list = [] 
        
        for dev_idx, test_idx in outer_kfold.split(self.X):        
            X_dev, X_test = self.X[dev_idx], self.X[test_idx]
            y_dev, y_test = self.y[dev_idx], self.y[test_idx]

            X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=VAL_SPLIT, shuffle = True, random_state=RANDOM_STATE)   
            
            ccnn = self.modelBuilder(batch_size)

            result = ccnn.train(X_train, y_train,
                                    stopping_rule=EarlyStoppingMonitor(0.1, 10, 1000),
                                    valid_X=X_val, valid_y=y_val)

            mse_val = result.data['valid_loss'][-1]
            print(mse_val)

            best_models_list.append(ccnn)

            y_pred = ccnn.predict(X_test)
            y_pred = y_pred[0]
            
            MSE_test = tools.MEE_metric(y_test, y_pred)
            test_error_list.append(MSE_test)

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=VAL_SPLIT, shuffle = True, random_state=RANDOM_STATE)   

        for i in range(K):

            ccnn = best_models_list[i]

            result = ccnn.train(X_train, y_train,
                                    stopping_rule=EarlyStoppingMonitor(0.1, 10, 1000),
                                    valid_X=X_val, valid_y=y_val)

            #evaluation on training set just for print
            y_pred_tr = ccnn.predict(X_train)
            MEE_train = tools.MEE_metric(y_train, y_pred_tr)

            y_pred = ccnn.predict(X_val)
            y_pred = y_pred[0]
            
            MEE_val = tools.MEE_metric(y_val, y_pred)
                                
            val_MEE_list.append(MEE_val)
            if (MEE_val == min(val_MEE_list)):
                best_training_error = MEE_train
                best_model_MEE_val = MEE_val #val_error
                best_model = ccnn #model

        mean_test_error = round(stats.mean(test_error_list),2)
        stdev_test_error = round(stats.stdev(test_error_list),2)
    
        return best_model, best_model_MEE_val, ccnn.num_hidden, mean_test_error, stdev_test_error, best_training_error



#CASCADE CORRELATION for classification
    def train_CC_clf(self, batch_size):

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=VAL_SPLIT, shuffle = True, random_state=RANDOM_STATE)   
        
        ccnn = self.modelBuilder(batch_size)

        result = ccnn.train(X_train, y_train,
                                stopping_rule=EarlyStoppingMonitor(0.1, 10, 1000),
                                valid_X=X_val, valid_y=y_val)

        mse_val = result.data['valid_loss'][-1]

        print(self.X.shape)
        print(self.y.shape)
        


        y_pred_training = ccnn.predict(self.X)
        y_pred_training = y_pred_training[0]
        
        MSE_training = mean_squared_error(self.y, y_pred_training)
        y_pred_training = np.rint(y_pred_training)
        accuracy_training = accuracy_score(self.y, y_pred_training)

        y_pred = ccnn.predict(self.X_test)
        y_pred = y_pred[0]

        best_model_MSE_TEST = mean_squared_error(self.y_test, y_pred)

        y_pred = np.rint(y_pred)

        accuracy_test = accuracy_score(self.y_test, y_pred)

        return ccnn, MSE_training, accuracy_training, best_model_MSE_TEST, accuracy_test, ccnn.num_hidden



    #COMMON TRAIN FUN (USED BY BOTH CLASS & REG)
    #perform a Bayesian Optimized search over a set of hyperp and return the best combination (with best val error)
    def train_standard(self, X_train, y_train, X_test, y_test):
        tuner = kt.BayesianOptimization(self.modelBuilder,
                             objective=self.metric,
                             max_trials=self.tunerParameters['max_trials'],
                             overwrite=True,
                             seed=round(RANDOM_STATE),
                             directory=self.tunerParameters['directory'],
            project_name=self.tunerParameters['project_name'])

        tuner.search(X_train, y_train, epochs=EPOCHS, validation_split=VAL_SPLIT, batch_size=self.tunerParameters['batch_size'], 
                    callbacks=self.callbacks, verbose=0)
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        

        model = tuner.hypermodel.build(best_hps)


        history = model.fit(X_train,y_train,validation_split=VAL_SPLIT, 
                    epochs=EPOCHS, batch_size=self.tunerParameters['batch_size'], verbose=0, callbacks = self.callbacks
                    )
                    
        
        y_pred = model.predict(X_test)

        if(self.mode in ['clf','classification']):
            test_error = mean_squared_error(y_test, y_pred)
        else:
            test_error = tools.MEE_metric(y_test, y_pred) 

        return best_hps, test_error, tuner

    #SPECIFIC TRAIN FUN, USED ONLY BY REG (PERFORMS K-FOLD)
    def train_regression(self):
        outer_kfold = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)

        val_MEE_list=[]      
        best_hps_list = []
        test_error_list = []

        print("############################")
        print("START OF OUTER K-FOLD")


        for dev_idx, test_idx in outer_kfold.split(self.X):
            
            X_dev, X_test = self.X[dev_idx], self.X[test_idx]
            y_dev, y_test = self.y[dev_idx], self.y[test_idx]
            best_hps, MEE_test, tuner = self.train_standard(X_dev, y_dev, X_test, y_test)
            best_hps_list.append(best_hps)
            test_error_list.append(MEE_test)
            
            print("-------------------")

        print("END OF OUTER K-FOLD")
        print("############################")


        #RE-DEFINE THE MODEL WITH EMPTY HYPERP
        print("############################")
        print("START FINAL MODEL SELECTION (done with hold-out)")

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=VAL_SPLIT, random_state=RANDOM_STATE)

        for i in range(K):

            model = tuner.hypermodel.build(best_hps_list[i])


            history = model.fit(X_train,y_train, validation_split=VAL_SPLIT,epochs=EPOCHS, 
                        batch_size=self.tunerParameters['batch_size'], verbose=0, callbacks = self.callbacks
                        )


            #evaluation on training set just for print
            y_pred_tr = model.predict(X_train)
            MEE_train = tools.MEE_metric(y_train, y_pred_tr)

            print("model fit test")            
            y_pred = model.predict(X_val)
            
            MEE_val = tools.MEE_metric(y_val, y_pred)

            #check the val_loss and save the best model so far
            val_MEE_list.append(MEE_val)
            if (MEE_val == min(val_MEE_list)):
                best_training_error = MEE_train
                best_model_MEE_val = MEE_val 
                best_params = best_hps_list[i] 
                history = history
                best_model = model 

            print("------------------")  
        tools.plot(history=history, mode=self.mode, name=self.modelName)
        print("END FINAL MODEL SELECTION")
        print("############################")

        mean_test_error = round(stats.mean(test_error_list),2)
        stdev_test_error = round(stats.stdev(test_error_list),2)
    
        return best_model, best_model_MEE_val, best_params, mean_test_error, stdev_test_error, best_training_error
    
