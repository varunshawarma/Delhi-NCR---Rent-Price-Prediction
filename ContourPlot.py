import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns


data = pd.read_csv('/Users/vs161/Desktop/EE Data/Data.csv')

#Remove NaN
for _ in data.columns:
    data[_].fillna(data[_].median(), inplace=True)

#Remove Outliers
z = np.abs(stats.zscore(data))
data = data[(z <3).all(axis=1)]

y = data['Price']
X = np.column_stack((data['Area'], data['Bathroom']))

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

def gradient_descent(W, x, y):
    y_hat = x.dot(W).flatten()
    error = (y - y_hat)
    mse = (1.0 / len(x)) * np.sum(np.square(error))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse


w = np.array((-40, -40))
alpha = .01
tolerance = 1e-3

old_w = []
errors = []

# Perform Gradient Descent
iterations = 1
for i in range(200):
    gradient, error = gradient_descent(w, X_scaled, y)
    new_w = w - alpha * gradient

    # Print error every 10 iterations
    if iterations % 10 == 0:
        print("Iteration: %d - Error: %.4f" % (iterations, error))
        old_w.append(new_w)
        errors.append(error)

    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print('Gradient Descent has converged')
        break

    iterations += 1
    w = new_w

print('w =', w)

all_ws = np.array(old_w)

# Just for visualization
errors.append(600)
errors.append(500)
errors.append(400)
errors.append(300)
errors.append(225)

levels = np.sort(np.array(errors))

w0 = np.linspace(-w[0] * 5, w[0] * 5, 100)
w1 = np.linspace(-w[1] * 5, w[1] * 5, 100)
mse_vals = np.zeros(shape=(w0.size, w1.size))

for i, value1 in enumerate(w0):
    for j, value2 in enumerate(w1):
        w_temp = np.array((value1,value2))
        mse_vals[i, j] = gradient_descent(w_temp, X_scaled, y)[1]

plt.contourf(w0, w1, mse_vals, levels, alpha=.7)
plt.axhline(0, color='black', alpha=.5, dashes=[2, 4], linewidth=1)
plt.axvline(0, color='black', alpha=0.5, dashes=[2, 4], linewidth=1)
for i in range(len(old_w) - 1):
    plt.annotate('', xy=all_ws[i + 1, :], xytext=all_ws[i, :],
                 arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                 va='center', ha='center')

CS = plt.contour(w0, w1, mse_vals, levels, linewidths=1, colors='black')
plt.clabel(CS, inline=1, fontsize=8)
plt.title("Contour Plot of Gradient Descent")
plt.xlabel("w0")
plt.ylabel("w1")
plt.show()