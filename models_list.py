

from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

from keras.layers import Dense, Input, Dropout
import keras
from keras import Model
import tensorflow as tf

from keras.optimizers import Adam, SGD

import time

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# KNN
from sklearn.neighbors import KNeighborsRegressor

# multioutput
from sklearn.multioutput import MultiOutputRegressor


#from sklearn.svm import SVM
from sklearn.linear_model import SGDClassifier, LinearRegression

from keras.layers import Dense, Input, Dropout
import keras
from keras import Model
import tensorflow as tf

from keras.optimizers import Adam

import time

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# KNN
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

#seed for replicability
RANDOM_STATE = 42

#set the hyperp defined by the user
def set_hyperp(hyperparam):
    global hyperp
    hyperp = hyperparam


#input size set accordingly to the dataset (17 for clf, 9 for regression)
def set_input_size(input_size_par):
    global input_size
    input_size = input_size_par



#######################--REGRESSION--#######################


#SKLEARN MODEL DEFINITION

#knn
def build_KNN_Pipe_Reg():
    scaler = StandardScaler()
    knn = KNeighborsRegressor()
    pipe = Pipeline([('scaler', scaler), ('model', MultiOutputRegressor(knn))])
    return pipe
    
#support vector regressor
def build_SVR(max_iter):
    svr = SVR(max_iter=max_iter) 
              #self.trainingParameters['epochs'])
    regressor = MultiOutputRegressor(svr)
    return regressor

#ridge regressor
def build_RidgeRegressor(max_iter):
    regressor =  MultiOutputRegressor(SGDRegressor(max_iter=max_iter))
    return regressor

#linear regressor
def build_Linear(max_iter):
    regressor = MultiOutputRegressor(LinearRegression(max_iter=max_iter))
    return regressor





#KERAS MODEL DEFINITION

#deep NN
def get_deepNN(hp):
    initializer = tf.keras.initializers.HeUniform(seed=RANDOM_STATE)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_size,)))

    hp_units = hp.Choice('units',hyperp['units'])
    hp_dropout = hp.Choice('dropout', hyperp['dropout'])
    hp_depth = hp.Int('depth', min_value=hyperp['depth'][0], max_value=hyperp['depth'][1], step=hyperp['depth'][2])
    hp_learning_rate = hp.Choice('learning_rate', values=hyperp['learning_rate'])
    hp_decay = hp.Choice('decay', values=hyperp['decay'])

    for i in range(hp_depth):
        model.add(Dropout(hp_dropout))
        model.add(Dense(hp_units, activation=hyperp['activation_hidden'], kernel_initializer=initializer))

    model.add(keras.layers.Dense(hyperp['output_units'],activation = hyperp['activation_output'], kernel_initializer=initializer))
    model.compile(loss='MSE', optimizer=Adam(learning_rate=hp_learning_rate, decay=hp_decay), metrics=hyperp['metric']) 
    return model

#convolutional NN
def get_CNN(hp):
    initializer = tf.keras.initializers.HeUniform(seed=RANDOM_STATE)
    n_steps = input_size #is the len of each single record
    n_features = 1 #is the num of different record to be considered as input 
    filters = 64 #is the num of different kernel used

    hp_learning_rate = hp.Choice('learning_rate', values=hyperp['learning_rate'])
    hp_decay = hp.Choice('decay', values=hyperp['decay'])
    hp_units = hp.Choice('units',hyperp['units'])

    inputs=Input(shape=(n_steps,n_features))
    Conv = keras.layers.Conv1D(filters=9, kernel_size=2, activation=hyperp['activation_hidden'], 
                                input_shape=(n_steps,n_features), kernel_initializer=initializer)(inputs)
    Pool = keras.layers.MaxPooling1D(pool_size=2, padding="valid")(Conv)

    Conv2 = keras.layers.Conv1D(filters=3, kernel_size=2, activation=hyperp['activation_hidden'],
                                kernel_initializer=initializer)(Pool)
    Pool2 = keras.layers.MaxPooling1D(padding='same')(Conv2)

    Fla = keras.layers.Flatten()(Pool2)
    hidd1 = keras.layers.Dense(units = hp_units, activation=hyperp['activation_hidden'],kernel_initializer=initializer)(Fla)
    hidd2 = keras.layers.Dense(hyperp['output_units'], activation=hyperp['activation_output'], kernel_initializer=initializer)(hidd1)
    model= Model(inputs=inputs, outputs=hidd2)

    model.compile(loss='MSE', optimizer=Adam(learning_rate=hp_learning_rate, decay=hp_decay),metrics=hyperp['metric'])    
    return model

#random NN
def get_RandNN(hp):
    n_features_in_ = input_size
    initializer = tf.keras.initializers.HeUniform(seed=RANDOM_STATE)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))

    hp_units = hp.Choice('units',hyperp['units'])
    hp_dropout = hp.Choice('dropout', hyperp['dropout'])
    hp_depth = hp.Int('depth', min_value=hyperp['depth'][0], max_value=hyperp['depth'][1], step=hyperp['depth'][2])
    hp_learning_rate = hp.Choice('learning_rate', values=hyperp['learning_rate'])
    hp_decay = hp.Choice('decay', values=hyperp['decay'])

    for i in range(hp_depth):
        model.add(Dropout(hp_dropout))
        model.add(Dense(hp_units, activation=hyperp['activation_hidden'],kernel_initializer=initializer, trainable=False))

    model.add(keras.layers.Dense(hyperp['output_units'],activation=hyperp['activation_output'], kernel_initializer=initializer))
    model.compile(loss='MSE', optimizer=Adam(learning_rate=hp_learning_rate, decay=hp_decay),
                     metrics=hyperp['metric'])
    return model

#cascade correlation NN   
def get_CC_units(batch_size):
    import tensorflow.compat.v1 as tf #importa l'optimizer utile al candidate unit
        #cascorr
    from cascor import activations, losses
    from cascor.model import CCNN
    from cascor.monitor import EarlyStoppingMonitor
    from cascor.units.perceptron import TensorflowPerceptron, ScipyPerceptron

    if(input_size==9):
        output_size = 2
        act_fun_output = [activations.linear] 
    else:
        output_size = 1
        act_fun_output = [activations.generalized_sigmoid]

    output_unit = TensorflowPerceptron(activations=act_fun_output, loss_function=losses.mse, stopping_rule=EarlyStoppingMonitor(patience=20, min_delta = 0.001, max_iters=2000, normalize=True))


    candidate_unit = TensorflowPerceptron(activations=[tf.nn.relu],
                                              loss_function=losses.mse,
                                              stopping_rule=EarlyStoppingMonitor(patience=20, min_delta = 0.1,  max_iters=2000, normalize=True),
                                              optimizer=tf.train.AdamOptimizer,
                                              optimizer_args={'learning_rate': 0.01},
                                              batch_size=batch_size)


    ccnn = CCNN(input_size, output_size, 
                    output_unit=output_unit, candidate_unit=candidate_unit,
                    metric_function=losses.fvu,
                    lambda_param=0.8)
        
    return ccnn





#######################--CLASSIFICATION--#######################

#SKLEARN MODEL DEFINITION


#knn
def build_KNN_Pipe_Clf():
    scaler = StandardScaler()
    knn = KNeighborsClassifier()
    pipe = Pipeline([('scaler', scaler), ('model',knn)])
    return pipe

#support vector machine    
def build_SVM(max_iter):
    svm = SVC(max_iter=max_iter) 
              #self.trainingParameters['epochs'])
    return svm

#ridge classifier
def build_RidgeClassifier(max_iter):
    classifier = SGDClassifier(max_iter=max_iter)
    return classifier

