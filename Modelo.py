# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:33:04 2023

@author: alejandrs
"""

import pandas as pd

bd=pd.read_csv('bd.csv')

from PIL import Image
i=Image.open('descarga.png','r') # imagen en color 
i.show()

"""-----Modelado-----"""
#Se inicia realizando un SGDClassifier siguiendo el mapa insertado anteriormente

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

import funciones as fd

arreglo = bd.values
X=arreglo[:,:-1]
y=arreglo[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) #separamos los datos

#Creamos el modelo con SGDClassifier

modelo1 = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))

modelo1.fit(X_train, y_train)

#sacamos las predicciones para x_test

y_pred1=modelo1.predict(X_test)

#Creamos el modelo linear lienar SVC 

modelo2= make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
modelo2.fit(X_train, y_train)
y_pred2=modelo2.predict(X_test)

modelo3=KNeighborsClassifier(n_neighbors=3)
modelo3.fit(X_train, y_train)
y_pred3=modelo3.predict(X_test)

modelo4= RandomForestClassifier(n_estimators=100, bootstrap = True, verbose=2, max_features = 'sqrt')
modelo4.fit(X_train, y_train)
y_pred4=modelo4.predict(X_test)


"""---Metricas---"""

fd.metricas(modelo1,X_train,y_train,X_test,y_test,y_pred1)
#El modelo acertó en un 85% con los datos de entrenamiento
#El modelo acertó en un 83% con los datos de prueba
#El modelo predijo personas que renunciarían y si lo hicieron en un 93.9%, pero predijo personas que renunciarían y no lo hicieron en un 24.5%
#El modelo predijo personas que renunciarían y si lo hicieron en un 86.5%, pero predijo personas que renunciarían y si lo hicieron en un 39.5%

fd.metricas(modelo2,X_train,y_train,X_test,y_test,y_pred2)
#El modelo acertó en un 87% con los datos de entrenamiento
#El modelo acertó en un 85% con los datos de prueba
#El modelo predijo personas que renunciarían y si lo hicieron en un 99.3%, pero predijo personas que renunciarían y no lo hicieron en un 7.01%
#El modelo predijo personas que renunciarían y si lo hicieron en un 85.2%, pero predijo personas que renunciarían y si lo hicieron en un 66.6%

fd.metricas(modelo3,X_train,y_train,X_test,y_test,y_pred3)
#El modelo acertó en un 88% con los datos de entrenamiento
#El modelo acertó en un 81% con los datos de prueba
#El modelo predijo personas que renunciarían y si lo hicieron en un 95.7%, pero predijo personas que renunciarían y no lo hicieron en un 4.8%
#El modelo predijo personas que renunciarían y si lo hicieron en un 84.4%, pero predijo personas que renunciarían y si lo hicieron en un 17.1%

fd.metricas(modelo4,X_train,y_train,X_test,y_test,y_pred4)
#El modelo acertó en un 100% con los datos de entrenamiento
#El modelo acertó en un 89% con los datos de prueba
#El modelo predijo personas que renunciarían y si lo hicieron en un 99%, pero predijo personas que renunciarían y no lo hicieron en un 35%
#El modelo predijo personas que renunciarían y si lo hicieron en un 89.2%, pero predijo personas que renunciarían y si lo hicieron en un 87%

"""----Afinamiento de hiperparametros---"""
