# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:42:15 2020

@author: alber
"""

# Regresión Lineal Múltiple 

# Importar librerías
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

# Dividir el dataset en conjunto de entrenamientoy conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables (Normalización)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjuto de testing
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

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
# Agregar una columna de 1´s y asociarlo como coeficiente al término independiente de la Regresión
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, aXis = 1)
SL = 0.05

# Se ha añadido el modificador .tolist() al X_opt para adaptarse a Python 3.7

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, eXog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, eXog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, eXog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, eXog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, eXog = X_opt.tolist()).fit()
regression_OLS.summary()
