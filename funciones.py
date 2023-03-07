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
    import pandas as pd
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

def imputar_f (df,list_cat):  
        
    from sklearn.impute import SimpleImputer
    import pandas as pd
    df_c=df[list_cat]
    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer(strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)

    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)

    df =pd.concat([df_n,df_c],axis=1)
    return df


def preparar_datos (df):
    import joblib
    import pandas as pd

    #######Cargar y procesar nuevos datos ######
   
    
    #### Cargar modelo y listas 
    
   
    list_cuali=joblib.load("list_cuali.pkl")
    list_dummies=joblib.load("list_dummies.pkl")
    var_names=joblib.load("var_names.pkl")
    
    ####Ejecutar funciones de transformaciones
    
    df=imputar_f(df,list_cuali)
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    df_dummies= df_dummies.loc[:,~df_dummies.columns.isin(['perf_2023','EmpID2'])]
    X=pd.DataFrame(df_dummies.values,columns=df_dummies.columns)
    X=X[var_names]
    
    
    
    
    return X