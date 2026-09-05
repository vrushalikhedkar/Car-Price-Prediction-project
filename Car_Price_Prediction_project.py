import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

dataset = pd.read_csv('car_price.csv')

dataset

dataset.shape

dataset.isnull().sum()

plt.scatter(dataset['Engine_Size'], dataset['Price'],cmap='rainbow')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.title('Engine Size vs Car Price')
plt.show()

X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

X

y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

print("Actual:", y_pred)
print("Predicted:", y_pred)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)

plt.scatter(y_test, y_pred, color='blue', label='Predicted')
plt.scatter(y_test, y_test, color='red', label='Actual')
plt.xlabel('Price')
plt.ylabel('Price')
plt.title('Actual vs Predicted Car Price')
plt.legend()
plt.show()

model.predict([[5, 2.0, 150, 2018]])


