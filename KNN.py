#Importing the required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

#Importing data
df = pd.read_csv('/Users/vs161/Desktop/EE Data/Data.csv')

#Filling in NaN/Missing values
for _ in df.columns:
    df[_].fillna(df[_].median(), inplace=True)

# Log Transformation
for _ in df.columns:
    df[_] = np.log(df[_])

#Removing Outliers using Z-Score
z = np.abs(stats.zscore(df))
df = df[(z <3).all(axis=1)]

#Splitting into training and test datasets
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.3)

x_train = train.drop('Price', axis=1)
y_train = train['Price']

x_test = test.drop('Price', axis = 1)
y_test = test['Price']


rmse_val = [] #to store rmse values for different k
K=2
while K in range(2,50):
    #Model for Weighted KNN Regression
    model = neighbors.KNeighborsRegressor(n_neighbors = K, weights= 'distance')
    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    K = K + 2

#Plotting the Elbow curve to find optimum k
curve = pd.DataFrame(rmse_val)
curve.plot()
plt.show()



model = neighbors.KNeighborsRegressor(n_neighbors = K, weights= 'distance')
model.fit(x_train, y_train)
pred=model.predict(x_test) #make prediction on test set

print('R2 score = ',r2_score(y_test,model.predict(x_test)))

sns.regplot(x=y_test, y=pred, ci=None, color="b")
plt.show()




