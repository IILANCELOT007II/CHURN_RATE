# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1= LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2= LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categories=[1], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],remainder='passthrough')
X = ct.fit_transform(X)

#X = onehotencoder.fit_transform(X, y = None).toarray()

#evading dummy variable trap
X = X[:, 1:]



#missing data
'''from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (THE MOST IMPORTANT PART IN DL)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Fitting classifier to the Training set
# Create your classifier here
classifier = Sequential()

#input and first hidden layer (RECTIFIER FUNCTION)
#hidden layer nodes = avg of input and output
classifier.add(Dense(6, init = 'uniform', activation = 'relu',input_dim = 11))

#second hidden layer
classifier.add(Dense(6, init = 'uniform', activation = 'relu'))

#Output layer (SIGMOID FUNCTION)
classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

#Compiler layer 
#Argument: Optimizer-Algorithm to find optimal weights
#Optimizer - ADAM
#Logarithmic loss function : if binary outcome = binary_crossentropy
#                            more than two outcomes= categorical_crossentropy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
#Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

#IMPROVISATION
X = np.array(X, dtype=np.float64)

#to check prediction for given data
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction>0.5)

cm = confusion_matrix(y_test, y_pred)


#Evaluation of ANN using K fold cross validation technique
# K = 10
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Since KerasClassifier requires a function argument so we define a function
#RELU = rectified linear unit   
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, init = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()

#Dropout regularization to reduce overfitting if needed
#EG: classifier.add(Dropout(0.1)) i.e 1 neuron will be dropped while training    


#Tuning the ann model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV 

#Since KerasClassifier requires a function argument so we define a function
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, init = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
parameter = {'batch_size':[16,32], 'epochs':[200,500], 'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameter, scoring = 'accuracy',  cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_


















