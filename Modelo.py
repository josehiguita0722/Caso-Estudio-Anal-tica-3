# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:33:04 2023

@author: alejandrs
"""

import pandas as pd

bd=pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/bd.csv')

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
import joblib
import funciones as fd
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

arreglo = bd.values
X1=bd.iloc[:,:-1]
y1=bd.iloc[:,-1]
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
modelos=[modelo4]

#DataFrame de resultados
cols=["Case",'RandomForestClassifier']
result= pd.DataFrame(columns=cols)
result.set_index("Case",inplace=True)
result.loc['Standard'] = [0]
result.loc['GridSearch'] = [0]
result.loc['RandomSearch'] = [0]

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


#RandomForest
n_estimators = [10, 100, 1000]
class_weight=["balanced", "balanced_subsample",None]
max_features = ['sqrt', 'log2']
forest_grid=dict(n_estimators=n_estimators,class_weight=class_weight,max_features=max_features)



""""GridSearch"""
#Solo se corre una vez por costo computacional, eliminar comentario para correr 
#Ahora se aplica GridSearch para cada uno
grids=[forest_grid]

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
 #Solo se corre una vez por costo computacional, eliminar comentario para correr 

from sklearn.model_selection import RandomizedSearchCV

col = 0
for ind in range(0,len(modelos)):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, 
                                 random_state=1)
    n_iter_search = 3
    random_search = RandomizedSearchCV(modelos[col],
    param_distributions=grids[col],n_iter=n_iter_search, cv=cv)
    random_search.fit(X_train,y_train)
    result.iloc[2,col] = random_search.score(X_test,y_test)
    col += 1
    
result.head()


""""Best Hyperparameters"""

best=grid_search.best_estimator_
print(best)

modelo5 = modelo5.fit(X_train,y_train)

""""Análisis modelo final"""

fd.metricas(modelo5,X_train,y_train,X_test,y_test,y_pred4)
# El porcentaje de acierto para el entrenemiento es del 100% lo que podría indicar un sobreajuste de los datos,
#El porcentaje de acierto de test es del 90% lo cuál se interpreta como un modelo aceptable
#La cantidad de personas que renunciarony y si lo hicieron fue del 99%, mientras que la cantidad e personas que se 
#predijo que renunciarian y no lo hicieron fue del 35%
#El porcentaje de personas que renunciaron y se predijo que lo harían fue del 89%, mientras que el porcentaje de personas que se 
#dijo que renunciarían y no lo hicieron fue del 86%

#####Evaluar métrica de entrenamiento y evaluación para mirar sobre ajuste ####
list_variables=joblib.load("list_variables.pkl")
eval1=cross_validate(best,X,y1,cv=5,scoring="neg_root_mean_squared_error",return_train_score=True) #base con todas las variables
eval2=cross_validate(modelo5,df3.iloc[:,:-1],df3.iloc[:,-1],cv=5,scoring="neg_root_mean_squared_error",return_train_score=True) #Base con variables significantes
#Los resultados el cross validate nos demuestran que el modelo no es  bueno, esto debe ser por el sobreajuste existente
#En el modelo, con esto se tiene que 

#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para RF
train_rf=pd.DataFrame(eval1['train_score'])
test_rf=pd.DataFrame(eval1['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']


#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para RL
train_rl=pd.DataFrame(eval2['train_score'])
test_rl=pd.DataFrame(eval2['test_score'])
train_test_rl=pd.concat([train_rl, test_rl],axis=1)
train_test_rl.columns=['train_score','test_score']

train_test_rl["test_score"].mean()
train_test_rf["test_score"].mean()


"""---Exportaciones---"""
joblib.dump(best,'best.pkl') #Se exportan los mejores hiperparámetros
joblib.dump(modelo5,'modelo5') #Se exporta el ajuste del modelo 

