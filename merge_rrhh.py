# -*- coding: utf-8 -*-
"""Merge RRHH.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mFrmzJLliXcfCSwDepEJVuh5XCEJ_bAb
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
import numpy as np
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor

df_retirement_info = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/df_renuncias.csv')
df_general_empleados = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/df_general_empleados1.csv')
df_desempeno = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/df_desempeno.csv')
df_out_time = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/df_out_time.csv')
df_in_time = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/df_in_time.csv')
df_encuesta_empleados = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/df_encuesta_empleados.csv')

data_rrhh = df_encuesta_empleados.merge(df_general_empleados,how='left',on='EmployeeID')
data_rrhh = data_rrhh.merge(df_desempeno,how='left',on='EmployeeID')
data_rrhh = data_rrhh.merge(df_in_time,how='left',on='EmployeeID').rename(columns={"horas": "hora_ingreso"})
data_rrhh = data_rrhh.merge(df_out_time,how='left',on='EmployeeID').rename(columns={"horas": "hora_salida"})
data_rrhh = data_rrhh.merge( df_retirement_info,how='left',on='EmployeeID')

data_rrhh = data_rrhh.fillna(0)

data_rrhh = data_rrhh.set_index('EmployeeID') #Convertimos el número de empleado en el índice

data_rrhh

#Al realizar el one hot, se identifica que Human Resources aparece cualitativamente en más de un feature.
data_rrhh.Department = data_rrhh.Department.replace({"Human Resources": 'Dept. Human Resources'})
data_rrhh.JobRole = data_rrhh.JobRole.replace({'Human Resources': 'Rol.Human Resources'})
data_rrhh.EducationField = data_rrhh.EducationField.replace({'Human Resources': 'Educ.Human Resources'})

"""## Label Encoding"""

ohencoder = OneHotEncoder(sparse= False) #Se carga el codigo necesario para poder convertir los datos cualitativos en cuantitativos

df = ohencoder.fit_transform(data_rrhh[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]) #Se ingresan las variables con datos cualitativos

labels = ohencoder.categories_ #Se obtienen los vectores con las categorias pertenecientes a cada variable
labels = np.concatenate(labels, axis=0 ) #Se unen

df = pd.DataFrame(df, columns = labels) #Se crean como data frame

df.head() #Se visualizan

df2 = data_rrhh[['EnvironmentSatisfaction', 'JobSatisfaction','WorkLifeBalance', 'Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
          'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement', 'PerformanceRating', 'hora_ingreso', 'hora_salida', 'Resignation']] #Se toman las variables cuantitativas

data_rrhh.dtypes

df3 = pd.concat([df.reset_index(drop=True), df2.reset_index(drop=True)], axis = 1) #Se unen las variables cuantitativas y las cualitativas transformadas a cuantitativas

df3.head(3)  # Visualización de data frame



"""# Feature Selection

##Método Univariate Selection (KBest)
"""

arreglo = df3.values #Se toman los valores del dataframe en formato numpy 
X = df3.iloc[:,:46].values
y = df3.iloc[:,46].values # Campo de interés Price

#crear un modelo de selección

est_prueba = SelectKBest(score_func=f_regression, k = 20)
est_ajustado = est_prueba.fit(X,y)

#Muestro el desempeño de los features basado en el valor F
np.set_printoptions(precision = 20,suppress = True)
#print(est_ajustado.scores_)
features = est_ajustado.transform(X)
#print(features)

temp2 = np.array(est_ajustado.scores_) #Genero un temporal con los puntajes de los estimadores ajustados

df4 = df3.copy()  #Se Muestra el puntaje K-Best asociado a cada variable en orden descendente para analizar las variables más representativas del modelo
df4.drop(columns = 'Resignation', inplace = True)
FS = pd.DataFrame({'Var':df4.columns,'KBest':temp2})
FS1=FS.sort_values(by = 'KBest', ascending = False).reset_index()
#hora salida, Single, TotalWorkingYears, 

"""##Recursive Feature Elimination"""

modelo = LinearRegression() #Se implementa el método RFE para elegir las variables más representativas del modelo
est_rfe = RFE(modelo, n_features_to_select = 20)
est_ajustado = est_rfe.fit(X,y)
print(est_ajustado.n_features_)
print(est_ajustado.support_)
print(est_ajustado.ranking_)

df4 = df3.copy() #Se muestran las variables seleccionadas por el método RFE
df4.drop(columns = 'Resignation', inplace = True)
temp2 = np.array((est_ajustado.support_))
FS = pd.DataFrame({'Var':df4.columns,'RFE':temp2})
FS2=FS[FS['RFE'] == True].reset_index()

"""##Feature Importance"""

# Se seleccionan las variables más significativas para el modelo según la Extracción de features 
modelo = ExtraTreesRegressor(n_estimators=100)
modelo.fit(X, y)
print(modelo.feature_importances_)

temp3 = np.array(modelo.feature_importances_) #Se genera un temporal para los resultados del método

df4 = df3.copy() #Se imprimen las variables más significativas para el modelo según este método
df4.drop(columns = 'Resignation', inplace = True)
FS = pd.DataFrame({'Var':df4.columns,'values':temp3})
FS3=FS.sort_values(by = 'values', ascending = False).reset_index()

"""Comparamos los 3 métodos para encontrar las variables similares entre ellos"""
import funciones as fd
temp=fd.comparar(FS1['Var'],FS2['Var'])
features= fd.comparar(temp,FS3['Var'])
#Adicional a las features anteriores, se necesita agregar algunas variables que son parte de la misma categoría anteriormente cualitativa

bd=pd.DataFrame(data=df3[features])
bd['Medical']= df3['Medical']
bd['Marketing']= df3['Marketing']
bd['Technical Degree']= df3['Technical Degree']
bd['Life Sciences']= df3['Life Sciences']
bd['Other']=df3['Other']

#También se agregaran las variables que en los mérodos de selección obtuvieron los mejores puntajes

bd['hora_salida']=df3['hora_salida']
bd['TotalWorkingYears']=df3['TotalWorkingYears']
bd['Age']=df3['Age']
bd['JobSatisfaction']=df3['JobSatisfaction']
bd['WorkLifeBalance']=df3['WorkLifeBalance']
bd['YearsWithCurrManager']=df3['YearsWithCurrManager']
bd['EnvironmentSatisfaction']=df3['EnvironmentSatisfaction']
bd['YearsSinceLastPromotion']=df3['YearsSinceLastPromotion']

