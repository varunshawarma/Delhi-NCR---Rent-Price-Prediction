import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
#from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from scipy import stats



data = pd.read_csv('/Users/vs161/Desktop/EE Data/Data.csv')

#Fill NAN values
for _ in data.columns:
#    if (_ == "Area"):
#        data[_].fillna(data[_].mean(), inplace=True)
    data[_].fillna(data[_].median(), inplace=True)
#data.fillna(0, inplace=True)
data = data.reset_index()

z = np.abs(stats.zscore(data))
print(np.where(z>3))
data = data[(z <3).all(axis=1)]

X = data[['Area', 'Bathroom']].values.reshape(-1,2)
Y = data['Price']

df2=pd.DataFrame(X,columns=['Area','Bathroom'])
df2['Price']=pd.Series(Y)


## Apply multiple Linear Regression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='Price ~ Area + Bathroom', data=df2)
results_formula = model.fit()
results_formula.params


## Prepare the data for Visualization

x_surf, y_surf = np.meshgrid(np.linspace(df2.Area.min(), df2.Area.max(), 100),np.linspace(df2.Bathroom.min(), df2.Bathroom.max(), 100))
onlyX = pd.DataFrame({'Area': x_surf.ravel(), 'Bathroom': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)



## convert the predicted result in an array
fittedY=np.array(fittedY)




# Visualize the Data for Multiple Linear Regression

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Area'],df2['Bathroom'],df2['Price'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Area')
ax.set_ylabel('Bathroom')
ax.set_zlabel('Price')
plt.show()