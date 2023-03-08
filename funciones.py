# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:10:36 2023

@author: alejandrs
"""
def comparar(a,b):
    vect=[]
    for i in a:
        for j in b:
            if i==j:
                vect.append(i)
    return vect

#Agregar en funciones lo de las horas del archivo caso_estudio

def metricas(modelo,X_train,y_train,X_test,y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import classification_report

    a= "train accuracy %.2f"%modelo.score(X_train,y_train) #0.85 El modelo se considera bueno,
    b=     "test accuracy  %.2f"%modelo.score(X_test,y_test) #0.83 #El modelo se considera bueno
    
    #Matriz de confusión
     #"La matriz de confusion es %.2f"%
    c=  confusion_matrix(y_test, y_pred)
    
    d= "El porcentaje de acierto al predecir es %.2f"%accuracy_score(y_test, y_pred) 
    
    e= recall_score(y_test, y_pred, average=None) #Verdaderos positivos y falsos negativos
    
    f= precision_score(y_test, y_pred, average=None) #verdaderos positivos y falsos positivos
    
    
    g= classification_report(y_test, y_pred, labels=[1]) #Intentar hacer este reporte en pandas
    
    return (a,b,
            "La matriz de confusion es",c,
            d,"True positive and false negative",e,
            "True positive and false positive",f,
            "Reporte de clasificación",g)



def preparar_datos (rrhh):
    import joblib
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder 
    import numpy as np
    #######Cargar y procesar nuevos datos ######
   
    
    #### Cargar modelo y listas 
    
   
    list_cuali=joblib.load("list_cuali.pkl")
    list_dummies=joblib.load("list_dummies.pkl")
    list_var=joblib.load("list_variables.pkl")
    cuanti=joblib.load('cuanti.pkl')
    
    ####Ejecutar funciones de transformaciones
    ohencoder = OneHotEncoder(sparse= False)
    df = ohencoder.fit_transform(rrhh[list_dummies])
    labels = ohencoder.categories_ #Se obtienen los vectores con las categorias pertenecientes a cada variable
    labels = np.concatenate(labels, axis=0 ) #Se unen
    df = pd.DataFrame(df, columns = labels)
    df2 = rrhh[cuanti]
    df3 = pd.concat([df.reset_index(drop=True), df2.reset_index(drop=True)], axis = 1)
    df3=df3[list_var]
    
    return df3