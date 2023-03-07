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

modelo1 =  SGDClassifier(max_iter=1000, tol=1e-3)

modelo1.fit(X_train, y_train)

#sacamos las predicciones para x_test

y_pred1=modelo1.predict(X_test)

#Creamos el modelo linear lienar SVC 

modelo2= LinearSVC(random_state=0, tol=1e-5)
modelo2.fit(X_train, y_train)
y_pred2=modelo2.predict(X_test)

#Modelo KNeighbors
modelo3=KNeighborsClassifier(n_neighbors=3)
modelo3.fit(X_train, y_train)
y_pred3=modelo3.predict(X_test)

#Modelo RandomForestClassifier
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

modelos=[modelo1,modelo2,modelo3,modelo4]

#DataFrame de resultados
cols=['Case','SGDClassifier','linear SVC','KNeighbors','RandomForestClassifier']
result= pd.DataFrame(columns=cols)
result.set_index("Case",inplace=True)
result.loc['Standard'] = [0,0,0,0]
result.loc['GridSearch'] = [0,0,0,0]
result.loc['RandomSearch'] = [0,0,0,0]
result.loc['Hyperopt'] = [0,0,0,0]

#Valores iniciales
col = 0
for model in modelos:
    model.fit(X_train,y_train)
    result.iloc[0,col] = model.score(X_test,y_test)
    col += 1

result.head()

#Metodo GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

#Creamos hiperparámetros para cada una de las distribuciones

#Los hiperparametros a cambiar se eligen en la sección de ayuda de cada uno de los modelos

#SGDClassifier
loss = ['hinge', 'modified_huber', 'log'] #Método
penalty = ['l1','l2'] #Penalidad de error
alpha= [0.0001,0.001,0.01,0.1] 
l1_ratio= [0.15,0.05,.025]
max_iter = [1,5,10,100,1000,10000]
sgd_grid = dict(loss=loss,penalty=penalty,max_iter=max_iter,alpha=alpha,l1_ratio=l1_ratio)

#Linear SCV
peanlty=['l1','l2']
loss = ['hinge', 'squared_hinge']
max_iter = [1,5,10,100,1000,10000]
scv_grid=dict(penalty=penalty, loss=loss, max_iter=max_iter)

#KNeighbors
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
knn_grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

#RandomForest
n_estimators = [10, 100, 1000,10000]
class_weight=["balanced", "balanced_subsample",None]
max_features = ['sqrt', 'log2']
forest_grid=dict(n_estimators=n_estimators,class_weight=class_weight,max_features=max_features)

""""GridSearch"""
#Ahora se aplica GridSearch para cada uno
grids=[sgd_grid,scv_grid,knn_grid,forest_grid]

col = 0

for ind in range(0,len(modelos)):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,random_state=1)
    grid_search = GridSearchCV(estimator=modelos[col], 
                  param_grid=grids[col], n_jobs=-1, cv=cv,  
                  scoring='accuracy',error_score=0)
    grid_clf_acc = grid_search.fit(X_train, y_train)
    result.iloc[1,col] = grid_clf_acc.score(X_test,y_test)
    col += 1

result.head()

""""RandomSearchCV"""

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(modelos[col], 
                param_distributions=grids[col], cv=cv)