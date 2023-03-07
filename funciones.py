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
    
    g= classification_report(y_test, y_pred)
    
    return (a,b,
            "La matriz de confusion es",c,
            d,"True positive and false negative",e,
            "True positive and false positive",f,
            "Reporte de clasificación",g)