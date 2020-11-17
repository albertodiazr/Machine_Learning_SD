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
train_df = pd.read_excel("Radio.xlsx")
score_df = pd.read_excel("Radio_Oct2020.xlsx")

dataset = pd.concat([train_df , score_df])

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

# Dividir el data set en conjunto de entrenamiento y conjunto de score
X_train = X[ 0:2407, : ]
X_score = X[ 2407: , : ]
y_train = y[ 0:2407 ]


# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 20, random_state = 0)
regression.fit(X_train, y_train)

# Predicción de nuestros modelos con Random Forest
y_pred = regression.predict(X_score)

# Redondear los resultados de la predicción
for i in range(0,len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0
y_pred = np.rint(y_pred)

# Resultados
print("Visitas totales de la predicción: " +str(sum(y_pred)))

# Exportar el resultado a Excel
df = pd.DataFrame(y_pred)
df.to_excel(excel_writer = "C:/Users/Alberto.dribon/Documents/Pruebas ML/Visitas/Radio_y_pred.xlsx")

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