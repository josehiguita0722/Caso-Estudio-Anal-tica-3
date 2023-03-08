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
    
    ####ver_predicciones_bajas ###
    emp_pred_bajo=perf_pred.sort_values(by=["pred"],ascending=True).head(10)
    
    emp_pred_bajo.set_index('EmpID2', inplace=True) 
    pred=emp_pred_bajo.T
    
    coeficientes=pd.DataFrame( np.append(m_lreg.intercept_,m_lreg.coef_) , columns=['coeficientes'])  ### agregar coeficientes
   
    pred.to_excel("prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    coeficientes.to_excel("coeficientes.xlsx") ### exportar coeficientes para analizar predicciones