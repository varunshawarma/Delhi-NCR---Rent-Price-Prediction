#Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#Importing data
data = pd.read_csv('/Users/vs161/Desktop/EE Data/Data.csv')

#Filling in NaN/Missing values
for _ in data.columns:
    data[_].fillna(data[_].median(), inplace=True)

# Log Transformation
for _ in data.columns:
    data[_] = np.log(data[_])

#Removing Outliers using Z-Score
z = np.abs(stats.zscore(data))
data = data[(z <3).all(axis=1)]

#Assigning X and Y values
x = data.iloc[:, 0:2]
y = data.iloc[:, 2]

#Standard Scaling (for Grad Descent)
sc = StandardScaler()
x = sc.fit_transform(x)

#Splitting into training and test datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size = 0.3,random_state=42)

#Defining the Cost function
def CostFunction(x, y, w, b):
    cost = np.sum((((x.dot(w) + b) - y) ** 2) / (2 * len(y)))
    return cost

#Defining the gradient descent algorithm
def GradientDescent(x, y, w, b, alpha, epochs):
    cost_list = [0] * epochs

    counter = 0
    for epoch in range(epochs):
        z = x.dot(w) + b
        loss = z - y

        weight_gradient = x.T.dot(loss) / len(y)
        bias_gradient = np.sum(loss) / len(y)

        w = w - alpha * weight_gradient
        b = b - alpha * bias_gradient

        cost = CostFunction(x, y, w, b)
        cost_list[epoch] = cost

        #if (epoch % (epochs /10) == 0):
        if (counter%5==0):
            print("Cost is:", cost)
        counter+=1

    return w, b, cost_list

w, b, c= GradientDescent(Xtrain, Ytrain, np.zeros(Xtrain.shape[1]), 0, 0.1,epochs=100)

#Plotting the Cost function convergence curve
plt.plot(c)
plt.show()

def predict(X, w, b):
    return X.dot(w) + b

y_pred = predict(Xtest, w, b)

y_t = predict(Xtrain, w, b)

#Defining the R2 score metric
def r2score(y_pred, y):
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

print("Train r2: ",r2score(y_t, Ytrain))
print("Test r2: ",r2score(y_pred, Ytest))
ax1 = sns.distplot(Ytest, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.show()
print("Assigned weights:")
print(w,b)