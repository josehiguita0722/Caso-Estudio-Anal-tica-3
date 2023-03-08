# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:39:44 2023

@author: ASUS
"""

import pandas as pd ### para manejo de datos
import joblib
import numpy as np
#!pip install openpyxl
import openpyxl
import funciones 

###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":


    ### conectarse a la base de datos ###
    rr_hh=joblib.load('data_rrhh.pkl')
 
  
    ####Otras transformaciones en python (imputación, dummies y seleccion de variables)
    df_t= funciones.preparar_datos(rr_hh)


    ##Cargar modelo y predecir
    modelo_4 = joblib.load("modelo4.pkl")
    predicciones=modelo_4.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['pred'])


    ###Crear base con predicciones ####
    rr_hh=rr_hh.reset_index()
    perf_pred=pd.concat([rr_hh['EmployeeID'],df_t,pd_pred],axis=1) 
    perf_pred.to_excel("Predicciones.xlsx") ### exportar coeficientes para analizar predicciones
    
