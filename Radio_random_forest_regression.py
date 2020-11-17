# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:22:53 2020

@author: alber
"""

# Regresión Bosques Aleatorios 

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_excel("Radio.xlsx")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values


# Codificar datos categóricos (Año, Mes, Dia semana y Hora)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1,2,3])],   
    remainder='passthrough')                      

X = onehotencoder.fit_transform(X)


# Evitar la trampa de las variables ficticias (dummy) - Eliminar la primera columna
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 20, random_state = 0)
regression.fit(X, y)

# Predicción de nuestros modelos con Random Forest
y_pred = regression.predict(X_test)

# Redondear los resultados de la predicción
for x in range(0,len(y_pred)):
    if y_pred[x] < 0:
        y_pred[x] = 0
y_pred = np.rint(y_pred)

# Contrastar los resultados
print("Visitas totales del test: " +str(sum(y_test)))
print("Visitas totales de la predicción: " +str(sum(y_pred)))
print("Diferencia Predicción VS Realidad: " + str(sum(y_pred)-sum(y_test)))
print("Error de la predicción (%): " +str((sum(y_pred)-sum(y_test))/sum(y_test)*100))


# Visualización de los resultados del Random Forest
"""
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión con Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
"""