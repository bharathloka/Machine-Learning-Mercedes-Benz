#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
import seaborn as sb

#importing dataset
#importing dataset
train = pd.read_csv("/Users/taduri/Downloads/train.csv")
test = pd.read_csv("/Users/taduri/Downloads/test.csv")
print(test.shape)
print(train.shape)

#dropping columns with constant values of train set
for col in train.columns:
    if len(train[col].unique())==1:
        train.drop(col,inplace=True, axis=1)

        
#dropping columns with constant values of test set
for col in test.columns:
    if len(test[col].unique())==1:
        test.drop(col,inplace=True, axis=1)
        
#Independent Variables of train set 
X = train.iloc[:, 2:386].values
#Independent Variables of train set 
test_X = test.iloc[:, 1:386].values
#Dependent Variable
y = train.iloc[:, 1]

#analysis of y value with indexes
plt.scatter(train.iloc[:,0],np.sort(y.values))
plt.xlabel('index')
plt.ylabel('y')
plt.show()

#analysis of y values range
sb.distplot(y, bins=50, kde=False)
plt.xlabel('y value')
plt.show()
print("Train shape : ", X.shape)
print("Test shape : ", y.shape)

datatypes = train.dtypes.reset_index()
datatypes.columns = ["Count", "Data Type"]
datatypes.groupby("Data Type").aggregate('count').reset_index()

#Encoding categorical data of train data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 
labelEncoder_0 = LabelEncoder()
X[:, 0] = labelEncoder_0.fit_transform(X[:, 0])

labelEncoder_1 = LabelEncoder()
X[:, 1] = labelEncoder_1.fit_transform(X[:, 1])
 
labelEncoder_2 = LabelEncoder()
X[:, 2] = labelEncoder_2.fit_transform(X[:, 2])
 
labelEncoder_3 = LabelEncoder()
X[:, 3] = labelEncoder_3.fit_transform(X[:, 3])
 
labelEncoder_4 = LabelEncoder()
X[:, 4] = labelEncoder_4.fit_transform(X[:, 4])
 
labelEncoder_5 = LabelEncoder()
X[:, 5] = labelEncoder_5.fit_transform(X[:, 5])
 
labelEncoder_6 = LabelEncoder()
X[:, 6] = labelEncoder_6.fit_transform(X[:, 6])
 
labelEncoder_7 = LabelEncoder()
X[:, 7] = labelEncoder_7.fit_transform(X[:, 7])
for i in range(0,7):
    plt.hist(X[:,i])
    plt.suptitle(i)
    plt.show()
for j in range(0,8):
    print(np.unique(X[:,j]))
	
onehotencoder = OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7])
X = onehotencoder.fit_transform(X).toarray()

#Encoding categorical data of test data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 
labelEncoder_0 = LabelEncoder()
test_X[:, 0] = labelEncoder_0.fit_transform(test_X[:, 0])

labelEncoder_1 = LabelEncoder()
test_X[:, 1] = labelEncoder_1.fit_transform(test_X[:, 1])
 
labelEncoder_2 = LabelEncoder()
test_X[:, 2] = labelEncoder_2.fit_transform(test_X[:, 2])
 
labelEncoder_3 = LabelEncoder()
test_X[:, 3] = labelEncoder_3.fit_transform(test_X[:, 3])
 
labelEncoder_4 = LabelEncoder()
test_X[:, 4] = labelEncoder_4.fit_transform(test_X[:, 4])
 
labelEncoder_5 = LabelEncoder()
test_X[:, 5] = labelEncoder_5.fit_transform(test_X[:, 5])
 
labelEncoder_6 = LabelEncoder()
test_X[:, 6] = labelEncoder_6.fit_transform(test_X[:, 6])
 
labelEncoder_7 = LabelEncoder()
test_X[:, 7] = labelEncoder_7.fit_transform(test_X[:, 7])

onehotencoder = OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7])
test_X = onehotencoder.fit_transform(test_X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Applying PCA on train data set
from sklearn.decomposition import PCA 
pca = PCA(n_components = 25)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explainedVariance = pca.explained_variance_ratio_  #percentage of variance explained by each of the principal components extracted
print(explainedVariance)
X_train_pca.shape


#Applying PCA on test data set
from sklearn.decomposition import PCA 
pca = PCA(n_components = 25)
test_X_pca = pca.fit_transform(test_X)
explainedVariance = pca.explained_variance_ratio_  #percentage of variance explained by each of the principal components extracted
print(explainedVariance)

################
#XGBoost Model##
################
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score
from xgboost import XGBRegressor
xgr = XGBRegressor(max_depth = 5, n_estimators = 100)
xgr.fit(X_train_pca, y_train)
predictions = xgr.predict(X_test_pca)
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("Explained variance : ", explained_variance_score(y_test, predictions))
print("Mean absolute error : ", mean_absolute_error(y_test, predictions))
test_predictions = xgr.predict(test_X_pca)
print(test_predictions)

#Applying k-Fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = xgr, X = X_train, y = y_train, cv = 10)
print(max(scores))

#####################
## Bagging Model ####
#####################

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bagging = BaggingRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 10)
bagging = bagging.fit(X_train_pca, y_train)
predictions = bagging.predict(X_test_pca)
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("Explained variance : ", explained_variance_score(y_test, predictions))
print("Mean absolute error : ", mean_absolute_error(y_test, predictions))
test_predictions = bagging.predict(test_X_pca)
print(test_predictions)

#Applying k-Fold cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = bagging, X = X_train, y = y_train, cv = 10)
print(max(scores))


########################
## DecisionTree Model ##
########################


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score

dtr = DecisionTreeRegressor(max_depth= 6)
dtr = dtr.fit(X_train_pca, y_train)
predictions = dtr.predict(X_test_pca)
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("Explained variance : ", explained_variance_score(y_test, predictions))
print("Mean absolute error : ", mean_absolute_error(y_test, predictions))
test_predictions = dtr.predict(test_X_pca)
print(test_predictions)

#Applying k-Fold cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = dtr, X = X_train, y = y_train, cv = 10)
print(max(scores))

#########################
## NeuralNetwork Model ##
#########################

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

nn = MLPRegressor(hidden_layer_sizes=(10,10), learning_rate_init=0.01, early_stopping=False)
nn = nn.fit(X_train_pca, y_train)
predictions = nn.predict(X_test_pca)
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("Explained variance : ", explained_variance_score(y_test, predictions))
print("Mean absolute error : ", mean_absolute_error(y_test, predictions))
test_predictions = nn.predict(test_X_pca)
print(test_predictions)
#Applying k-Fold cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = nn, X = X_train, y = y_train, cv = 10)
print(max(scores))

##############################
## k-NearestNeighbours Model## 
##############################

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_pca, y_train)
predictions = knn.predict(X_test_pca)
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("Explained variance : ", explained_variance_score(y_test, predictions))
print("Mean absolute error : ", mean_absolute_error(y_test, predictions))
test_predictions = knn.predict(test_X_pca)
print(test_predictions)

#Applying k-Fold cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
print(max(scores))





####################################
########## Deep Learning ###########
####################################

#Importing the keras libraries and packages 
import keras
from keras.models import Sequential #To initialize the ANN (defining it as sequence of layers)
from keras.layers import Dense #To create layers in our ANN 

regressor = Sequential()

#Adding the input layer and first hidden layer
regressor.add(Dense(output_dim = 276, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))

#Adding the second hidden layer
regressor.add(Dense(output_dim = 276, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Adding the third hidden layer
regressor.add(Dense(output_dim = 276, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Adding the fourth hidden layer
regressor.add(Dense(output_dim = 276, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Adding output layer 
regressor.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'linear'))

#Compiling ANN (Applying stochastic gradient descent)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

#Visualize NN architecture
print(regressor.summary())

#Fitting the ANN to the training set 
regressor.fit(X_train_pca, y_train, validation_split=0.33, epochs=1000, batch_size=100)
test_predictions = regressor.predict(test_X_pca)
print(test_predictions)

