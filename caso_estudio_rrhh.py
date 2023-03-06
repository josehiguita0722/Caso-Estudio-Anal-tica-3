import pandas as pd #Manejo de datos
import numpy as np #Manejo de datos
import plotly.express as px #Gráficos
import matplotlib.pyplot as plt #Gráficos
from plotly.subplots import make_subplots #Gráficos

"""## Base de Datos Retiros"""

"""----------------------Cargamos las bd----------------"""

retirement_info = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/retirement_info.csv', sep=';')

"""-------------------Verificamos su correcta carga-----------------------"""

retirement_info
#LA base de datos cuenta con 23 atributos y su índice. Tiene datos de 4410 empleados

retirement_info.dtypes #Se observa el tipo de cada una de las variables en la base de datos

retirement_info['retirementDate']= pd.to_datetime(retirement_info['retirementDate']) #Cambiamos la variable a formato fecha
retirement_info.dtypes

retirement_info.columns #Se observan las columnas

retirement_info['retirementType'].unique()
# Observamos que hay dos tipos de retiro que son: Renuncias y despidos, en el caso de estudio solo nos interesa predecir las renuncias

#filtramos los datos que tengan despidos
retirement_info1 = retirement_info.loc[retirement_info['retirementType'] == 'Resignation']
#empleados.loc[empleados['Nombre'] == 'Juan'] 
retirement_info1.head()
#De esta manera obtenemos un dataframe solo con renuncias

retirement_info1

"""**Tratamiento de datos nulos**

"""

retirement_info1.isna().sum() #No se observan datos nulos

retirement_info1['resignationReason'].unique()

rt = retirement_info1.copy() #Realizamos una copia de la base de datos para trabajar en ella
renuncias=pd.get_dummies(rt['retirementType']) #Creamos una base de datos con valores dummy de las personas que renunciaron o se retiraron 
#del(renuncias['Fired']) #Eliminamos las personas que fueron despedidas ya que no son de interés 
renuncias['EmployeeID']=rt['EmployeeID'] #Agregamos el ID del empleado que tomo cada una de las desiciones
renuncias #Visualizamos

"""## Base de Datos General"""

"""----------------------Cargamos las bd----------------"""

general_empleados = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/general_data.csv', sep=';')

"""-------------------Verificamos su correcta carga-----------------------"""

general_empleados
#LA base de datos cuenta con 23 atributos y su índice. Tiene datos de 4410 empleados

general_empleados.columns #Se observan las columnas

general_empleados.dtypes #Se observa el tipo de cada una de las variables en la base de datos

"""**Tratamiento de datos nulos del dataset general_empleados**"""

general_empleados.isna().sum() #Se observan que solo hay datos nulos en el número de compañias en las que los empleados han trabajado y en los años trabajados en la compañia.

#se organiza los empleados por edad en orden de mayor a menor para aplicar un interpolated en los nulos de ambas columnas, 
#esto con el fin de que los años trabajados en la compañia y el numero de compañias rellenados tenga sentido.
empleados_por_edad = general_empleados.sort_values('Age',ascending=False)
empleados_por_edad.head(60)

#Al nuevo dataframe aplicamos le tratamiento de estos datos nulos con interpolated
# interpolate(): # va a interpolar los valores medios entre dos filas, nos dice cual seria el valor medio entre los registros (interpolación lineal)
general_empleados1 = empleados_por_edad.interpolate() # aplicar método
general_empleados1 = general_empleados1.sort_values('EmployeeID',ascending=False)  #Ordenamos el dataframe de acuerdo al ID como estaba inicialmente 
general_empleados1.isna().sum()  # verificamos si el dataset quedó con datos nulos

"""**Descripción de cada atributo cuantitativo**"""

"""------- Analizamos brevemente cada atributo cuantitativo -------"""

general_empleados1.describe()  
#Age: La media de edad de la empresa es de personas con experiencia, donde no se tienen personas menores de edad.
#DistanceFromHome	: El empleado que vive más lejos está a 29 km de su casa y el que está más cerca, está a 1 km de su casa.
#Education: El nivel de educación está en 1 y 5.
#EmployeeID: El ID de los empleados coincide con el número de los empleados.
#JobLevel:	Job level at company on a scale of 1 to 5.
#MonthlyIncome: la media del salario mensual de los trabajadores es de 65029 rupias, los empleados que ganan menos es de 10090 y el que más gana 199990 rupias.
#NumCompaniesWorked: La cantidad de empresas promedio en la que trabajan los empleados es de 2.694830	, 
#                    la empresa cuenta con empleados de primera experiencia y experimentados de 9 empresas.
#PercentSalaryHike: A todos los empleados se les aumentó el salario, el porcentaje promedio que se le aumentó fue de 15.209524 porciento.	
#StandardHours: Todos los empleados trabajan una jornada de 8 horas.
#TotalWorkingYears: Como era de esperarse hay empleados que no han cumplido el primer año de trabajo y hay algunos hasta con 40 años laborando.
#TrainingTimesLastYear: El máximo de capacitaciones al año son 6 probablemente a los empleados más nuevos y 0 capacitaciones a algunos empleados 
#                       que probablemente son los más antiguos.
#YearsAtCompany: Hay empleados que llevan 40 años en la empresa lo que indica que llevan toda su vida laborando en esta y otros que no llegan al año lo que indica que 
#                es su primera empresa en la que laboran.
#YearsSinceLastPromotion: Hay empleados que llevan 15 años sin ser promovidos
#YearsWithCurrManager: Como el máximo es 17 años, se deduce que el gerente actual lleva 17 años en el cargo.

general_empleados1['StockOptionLevel'].value_counts() #???
general_empleados1['TotalWorkingYears'].value_counts() #Hay más cantidad de empleados que llevan 10 años laborando. 
general_empleados1['YearsSinceLastPromotion'].value_counts() #La mayoría de empleados que no son promovidos son los que llevan menos tiempo
general_empleados1['NumCompaniesWorked'].value_counts()

"""**Observación de cada dato contenido en las variables categóricas**



"""

"""------- Buscamos tipos de datos en cada variable categórica-------"""

general_empleados1['Department'].unique() #Hay 3 departamentos 'Sales', 'Research & Development', 'Human Resources'
general_empleados1['BusinessTravel'].unique() #'Travel_Rarely', 'Travel_Frequently', 'Non-Travel'= 'Viaje_rara vez', 'Viaje_frecuentemente', 'No viajes'
general_empleados1['EducationField'].unique() # Se consideran 6 áreas de educación Life Sciences', 'Other', 'Medical', 'Marketing','Technical Degree', 'Human Resources'
general_empleados1['Gender'].unique() # Hombre y Mujer
general_empleados1['JobRole'].unique() # Hay 9 tipos de puestos de trabajos
general_empleados1['MaritalStatus'].unique() #'Married', 'Single', 'Divorced' = 'Casado', 'Soltero', 'Divorciado'
general_empleados1['Over18'].unique() #Todos los empleados son mayor de edad

"""**Gráficos de pastel**"""

# Gráfico de pastel CORREGIDO

# Creating plot
basegraficar = general_empleados1.groupby(['Department'])[['EmployeeCount']].count().sort_values('EmployeeCount', ascending = False).reset_index()
basegraficar.head()

fig = px.pie(basegraficar, values = basegraficar['EmployeeCount'],  names = basegraficar['Department'],
        title = '<b>Porcentaje de empleados por departamento<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

# Gráfico de pastel

# Creating plot

basegraficar = general_empleados1.groupby(['Department'])[['EmployeeCount']].count().sort_values('EmployeeCount', ascending = False).reset_index()
basegraficar.head()

fig = px.pie(general_empleados['Gender'].unique(), values = general_empleados['Gender'].value_counts(),  names = general_empleados['Gender'].unique(),
        title = '<b>Porcentaje de Hombres y Mujeres<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

"""## Base de Datos Encuesta de Desempeño de Empleados"""

"""----------------------Cargamos las Bases de datos----------------"""
desempeno = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/manager_survey_data.csv')
desempeno

"""---------------------Revisión de datos faltantes----------------"""
for i in desempeno.columns:
  a = desempeno[i].isna().sum()
  if a == 4410:
    del(desempeno[i])
  print(i,a)
#No se tiene datos nulos en la base de datos

desempeno = desempeno.set_index('EmployeeID') #Convertimos el número de empleado en el índice

print(desempeno['JobInvolvement'].unique()) 
print(desempeno['PerformanceRating'].unique()) #Solo se tienen desempeños excelentes y sobresalientes(3 y 4 respectivamente)

desempeno['PerformanceRating'].value_counts()

desempeno['JobInvolvement'].value_counts()

desempeno.dtypes

desempeno.boxplot() # La calificación de desempeño tiene un sesgo, cómo ya se había evidenciado sólo valoraciones de 3 y 4

# Gráfico de pastel

# Creating plot

fig = px.pie(desempeno['JobInvolvement'].unique(), values = desempeno['JobInvolvement'].value_counts(),  names = desempeno['JobInvolvement'].unique(),
        title = '<b>Nivel de participación en el trabajo<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

#Del grafico se observa que en un 59% de los empleados se tienen una participación alta(3)

# Creating plot

fig = px.pie(desempeno['PerformanceRating'].unique(), values = desempeno['PerformanceRating'].value_counts(),  names = desempeno['PerformanceRating'].unique(),
        title = '<b>Calificación de desempeño del año pasado<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

"""## Base de Datos Hora de Salida"""

"""----------------------Cargamos la Base de datos----------------"""
out_time = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/out_time.csv')
out_time
# La base de datos cuenta con 261 variables de fecha y el número de identificación de empleado. Tiene datos de 4410 empleados
# Cada variable hace referencia a la hora de salida en cada fecha de los operarios en el año 2015

out_time.rename(columns = {'Unnamed: 0':'EmployeeID'}, inplace = True)

"""------- Buscamos datos nulos en cada variable -------"""

for i in out_time.columns:
  a=out_time[i].isna().sum()
  print(i,a)
out_time

#Hay varios días en los que no asistió ningún empleado,  suponiendo que fue orden de la empresa la falta

out_time.shape
# Se eliminaron 12 columnas

"""-------Tipo de datos de las variables---"""
out_time.dtypes.value_counts()

#Todas son tipo objeto excepto por la primera. Así, teniendo en cuenta que la primera tiene el número de empleado se convierte en indice

out_time=out_time.fillna(0)

"""------Cambios-----"""

out_time=out_time.set_index(['EmployeeID']) #Se convierte la columna del número de empleado en el índice

for i in out_time.columns: #Se convierten los datos a tipo tiempo
  out_time[i] = pd.to_datetime(out_time[i],)

out_time=out_time.fillna(0)

out_time

df = out_time.copy()

bd = df.merge(renuncias,how='left',on='EmployeeID').fillna(0) #Unimos los datos sobre las renuncias con los de esta base de de datos y llenamos los datos inexistentes con 0
bd #Visualizamos

#Para el análsiis se podría estudiar en promedio a qué hora llegaban las personas que renunciaron y compararlo con el promedio de las demás personas
bd2= bd[bd['Resignation']==0]

#Necesitamos que el índice sea el empleado para poder transponerla y facilitar el análisis
bd=bd.set_index('EmployeeID')

#Transponemos
bd=bd.T

bd=bd.drop('Resignation',axis=0)

#Analizamos la forma correcta de extraer los datos que se necesitan

bd[2][0].hour
#bd[2][0].minute
#bd[2][0].second

""" --- ¿A qué hora llegaron en promedio los empleados que renunciaron? ---"""
horas1=[]
h=0
for i in bd.columns:
  for j in range(261):
    h += (bd[i][j].hour + (bd[i][j].minute/60 ) + (bd[i][j].second/120))
  horas1.append(h/262) #Creo que es mejor sacar la media, no el promedio
  h=0

bd1=pd.DataFrame(data = horas1, index=bd.columns, columns=['horas'] )

fig1=px.box(bd1) #En promedio los empleados que renunciaron tienen un rango de llegadas entre las 8:48 am y las 9:36 con un promedio de 9:13 aproximadamente
fig1.show()

"""----------------------Cargamos las bd----------------"""
in_time = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/in_time.csv')

in_time

"""-------------------Verificamos su correcta carga-----------------------"""

in_time 
in_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace= True)
in_time
#LA base de datos cuenta con 261 variables y su índice. Tiene datos de 4410 empleados
#Cada variable hace referencia a cada fecha en la que ingresan los operarios en el año 2015
#Se logra observar a simple vista que el 1 de Enero del 2015 aparentemente nadie trabajo, así que se podría borrar ese día. Además, como en las columnas se tiene el día se podría colocar en la tabla solo las horas o colocar todo en formato fecha

"""------- Buscamos datos nulos en cada variable -------"""

for i in in_time.columns:
  a=in_time[i].isna().sum()
  print(i,a)
in_time
#Hay varios días en los que no asistió ningún empleado, se supone que fue orden de la empresa la falta

in_time.shape
#Se eliminaron 12 columnas

"""-------Tipo de datos de las variables---"""
for i in in_time.columns:
  print(in_time[i].dtypes, i )
in_time=in_time.fillna(0)
#Todas son tipo objeto excepto por la primera que tiene el número de empleado

"""------Cambios-----"""
in_time=in_time.set_index(['EmployeeID']) #Se convierte la columna del número de empleado en el índice

for i in in_time.columns: #Se convierten los datos a tipo tiempo
  in_time[i]=pd.to_datetime(in_time[i],)

in_time=in_time.fillna(0)

in_time

in_time.iloc[0:,0]

in_time['2015-01-02'].dt.hour

"""---Se comprueban los cambios---"""

print(in_time.head(5))


in_time['2015-01-02'].dt.hour #Hora correcta
in_time['2015-01-02'].dt.minute #Minuto correcto 
in_time['2015-01-02'].dt.second #Segundo correcto

df=in_time.copy()

"""##Ahora con respecto a  la variable objetivo"""

bd=df.merge(renuncias,how='left',on='EmployeeID').fillna(0) #Unimos los datos sobre las renuncias con los de esta base de de datos y llenamos los datos inexistentes con 0
bd #Visualizamos

#Para el análsiis se podría estudiar en promedio a qué hora llegaban las personas que renunciaron y compararlo con el promedio de las demás personas
bd2= bd[bd['Resignation']==0]

#Necesitamos que el índice sea el empleado para poder transponerla y facilitar el análisis
bd=bd.set_index('EmployeeID')

#Transponemos
bd=bd.T

bd=bd.drop('Resignation',axis=0)

#Analizamos la forma correcta de extraer los datos que se necesitan

bd[2][0].hour
#bd[2][0].minute
#bd[2][0].second

""" --- ¿A qué hora llegaron en promedio los empleados que renunciaron? ---"""
horas1=[]
h=0
for i in bd.columns:
  for j in range(261):
    h+=(bd[i][j].hour + (bd[i][j].minute/60 ) + (bd[i][j].second/120 ))
  horas1.append(h/262) #Creo que es mejor sacar la media, no el promedio
  h=0

horas1 #Hora promedio en la que llegó cada empelado

bd1=pd.DataFrame(data = horas1, index=bd.columns, columns=['horas'] )

!pip install plotly
import plotly.express as px

fig1=px.box(bd1) #En promedio los empleados que renunciaron tienen un rango de llegadas entre las 8:48 am y las 9:36 con un promedio de 9:13 aproximadamente
fig1.show()

"""#Base de Datos Encuesta de Empleados(employee_survey_data.csv)"""

"""----------------------Cargamos las bd----------------"""

encuesta_empleados = pd.read_csv('https://raw.githubusercontent.com/josehiguita0722/Caso-Estudio-Anal-tica-3/main/employee_survey_data.csv')

"""-------------------Verificamos su correcta carga-----------------------"""

encuesta_empleados 
#LA base de datos cuenta con 3 variables y su índice. Tiene datos de 4410 empleados

"""------- Buscamos tipos de datos en cada variable -------"""

encuesta_empleados['EmployeeID'].unique()
encuesta_empleados['EnvironmentSatisfaction'].unique() #El campo tiene datos nulos y enteros
encuesta_empleados['EnvironmentSatisfaction'].isna().sum() #Se tienen 25 datos vacios
encuesta_empleados['JobSatisfaction'].unique() #El campo tiene datos nulos y enteros
encuesta_empleados['JobSatisfaction'].isna().sum() #El campo tiene datos nulos #Se tienen 20 datos vacios
encuesta_empleados['WorkLifeBalance'].unique() #El campo tiene datos nulos y enteros
encuesta_empleados['WorkLifeBalance'].isna().sum() #El campo tiene 38 datos vacios

encuesta_empleados[encuesta_empleados['EnvironmentSatisfaction'].isna()==True] #Los valores nulos en esta variable sólo son nulos en esta variable
encuesta_empleados[encuesta_empleados['JobSatisfaction'].isna()==True]#Los valores nulos en esta variable sólo son nulos en esta variable
encuesta_empleados[encuesta_empleados['WorkLifeBalance'].isna()==True]#Los valores nulos en esta variable sólo son nulos en esta variable

encuesta_empleados['EmployeeID'].dtype #Los datos en la variable columna son enteros
encuesta_empleados['EnvironmentSatisfaction'].dtype #Los datos de la columna son tipo float, no enteros
encuesta_empleados['JobSatisfaction'].dtype #Los datos de la columna son tipo float, no enteros
encuesta_empleados['WorkLifeBalance'].dtype #Los datos de la columna son tipo float, no enteros

"""----Se cambian datos nulos por la media ---""" #O se eliminan pues acá estoy en duda

#Este cambio se hace suponiendo que los empleados no quisieron llenar esa parte del formulario

a=encuesta_empleados['EnvironmentSatisfaction'].mean()
encuesta_empleados['EnvironmentSatisfaction']=encuesta_empleados['EnvironmentSatisfaction'].fillna(a) #Cambio
encuesta_empleados['EnvironmentSatisfaction'].isna().sum() #Se prueba

a=encuesta_empleados['JobSatisfaction'].mean()
encuesta_empleados['JobSatisfaction']= encuesta_empleados['JobSatisfaction'].fillna(a) #Cambio
encuesta_empleados['JobSatisfaction'].isna().sum()#Se prueba

a=encuesta_empleados['WorkLifeBalance'].mean()
encuesta_empleados['WorkLifeBalance']= encuesta_empleados['WorkLifeBalance'].fillna(a)#Cambio
encuesta_empleados['WorkLifeBalance'].isna().sum()#Se prueba

encuesta_empleados=encuesta_empleados.set_index('EmployeeID') #Convertimos el número de empleado en el índice

"""----Cambio de tipo de datos tipo float a int para disminuir costo computacional----"""

encuesta_empleados['EnvironmentSatisfaction']=encuesta_empleados[['EnvironmentSatisfaction']].astype('int32') #Se realiza el cambio de tipo de dato
encuesta_empleados['EnvironmentSatisfaction'].dtype  #Se comprueba el cambio

encuesta_empleados['JobSatisfaction']=encuesta_empleados[['JobSatisfaction']].astype('int32')#Se realiza el cambio de tipo de dato
encuesta_empleados['JobSatisfaction'].dtype # Se comprueba el cambio

encuesta_empleados['WorkLifeBalance']=encuesta_empleados[['WorkLifeBalance']].astype('int32')#Se realiza el cambio de tipo de dato
encuesta_empleados['WorkLifeBalance'].dtype # Se comprueba el cambio

encuesta_empleados.boxplot() #La satisfacción del ambiente y del trabajo se encuentran sesgadas y con un promedio de 3. 
#El balance entre en trebajo y la vida personal tiene su mediana en el extremo superior significando que el 75% de sus empleados 
#se sienten identificados con este valor. Adicionalmente, se observa que el valor mínimo es de 1 y el máximo es de 4

encuesta_empleados['WorkLifeBalance'].unique()

encuesta_empleados

#REalizar gráfico de pastel para ver en qué porcentaje están en cada categoria en cada variable.

fig = px.pie(encuesta_empleados['EnvironmentSatisfaction'].unique(), values = encuesta_empleados['EnvironmentSatisfaction'].value_counts(),  names = encuesta_empleados['EnvironmentSatisfaction'].unique(),
        title = '<b> Nivel de satisfacción con el ambiente de laboral<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

#Del grafico se logra inferir que en un 60% de los empleados se encuentran satisfechos con el ambiente laboral en la empresa. ESte es un valor que se puede mejorar por medio de un formulario hacia los 
#empleados sobre qué aspectos les gustaría mejorar en su ambiente laboral y también realizar ejercicios para fortalecer el trabajo en equipo

#REalizar gráfico de pastel para ver en qué porcentaje están en cada categoria en cada variable.

fig = px.pie(encuesta_empleados['JobSatisfaction'].unique(), values = encuesta_empleados['JobSatisfaction'].value_counts(),  names = encuesta_empleados['JobSatisfaction'].unique(),
        title = '<b>Nivel de satisfacción laboral<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

#Del gráfico se logra inferir que los valores más votados son el 4 y el 3. Esto significa, que el 70% de los empleados se encuentran satisfechos con su labor.

#REalizar gráfico de pastel para ver en qué porcentaje están en cada categoria en cada variable.

fig = px.pie(encuesta_empleados['WorkLifeBalance'].unique(), values = encuesta_empleados['WorkLifeBalance'].value_counts(),  names = encuesta_empleados['WorkLifeBalance'].unique(),
        title = '<b>Balance trabajo - vida personal<b>',
             color_discrete_sequence = px.colors.qualitative.G10
        )

fig.show()

#Del gráfico se logra inferir que los valores más votados son el 3 y 2. Este valor se da en un 84%, lo cuál significa que la empresa debería hacer algo al respecto ya que evidentemente 
#sus trabajadores tienen que sacrificar mucho en el momento de trabajar

"""##Con respecto a los retiros"""

bd=encuesta_empleados.merge(renuncias,how='left',on='EmployeeID').fillna(0) #Unimos los datos sobre las renuncias con los de esta base de de datos y llenamos los datos inexistentes con 0
bd #Visualizamos

bd=bd[bd['Resignation']==1] #Base de datos para analizar los resultados de solo las personas que renunciaron

#¿Cómo afecta el balance entre el trabajo y la vida personal en la satisfacción con el ambiente laboral? 
#¿Cómo afecta la satisfacción laboral al balance entre la vida y el trabajo?

"""---¿Cómo calificaron la satisfacción en la vida trabajo las personas que renunciaron?---"""

# crear dataset
base = bd.groupby(['WorkLifeBalance'])[['EmployeeID']].count().sort_values('WorkLifeBalance',ascending= False).reset_index().rename(columns={'EmployeeID':'Workers'})

# crear gráfica
fig = px.pie(base, values = 'Workers' , names = 'WorkLifeBalance',
             title= '<b> Sattisfacción del Balance vida-trabajo  <b>',
             color_discrete_sequence=px.colors.qualitative.G10)

# agregar detalles a la gráfica
fig.update_layout(
    template = 'simple_white',
    legend_title = '<b> Balance vida-tabajo <b>',
    title_x = 0.5)

fig.show()

#Las personas que renunciaron en un 54.1% calificaron la satisfacción de este campo con un valor de 3, significando que existe un amayor probabilidad de renuncia de las personas que que presentan este
#nivel de satisfacción que las personas que en realidad marcan que están inconformes con los valores como 1 o 2. 
#Cómo era de esperarse la mínoria de personas con la probabilidad de renuncia son aquellas que marcan el valor de 4 en satisfacción.

"""---¿Cómo calificaron la satisfacción del ambiente laboral las personas que renunciaron?---"""

# crear dataset
base = bd.groupby(['EnvironmentSatisfaction'])[['EmployeeID']].count().sort_values('EnvironmentSatisfaction',ascending= False).reset_index().rename(columns={'EmployeeID':'Workers'})

# crear gráfica
fig = px.pie(base, values = 'Workers' , names = 'EnvironmentSatisfaction',
             title= '<b> Sattisfacción del ambiente laboral  <b>',
             color_discrete_sequence=px.colors.qualitative.G10)

# agregar detalles a la gráfica
fig.update_layout(
    template = 'simple_white',
    legend_title = '<b> Ambiente laboral <b>',
    title_x = 0.5)

fig.show()

#No se logra apreciar una diferencia significativa de las personas que renunciaron y marcaron determinado valor en esta sección. El mayor porcentaje de personas que renunciaron marcó un valor de 1
# y el de menor porcentaje el que marco un valor de 2.

"""---¿Cómo calificaron la satisfacción laboral las personas que renunciaron?---"""

# crear dataset
base = bd.groupby(['JobSatisfaction'])[['EmployeeID']].count().sort_values('JobSatisfaction',ascending= False).reset_index().rename(columns={'EmployeeID':'Workers'})

# crear gráfica
fig = px.pie(base, values = 'Workers' , names = 'JobSatisfaction',
             title= '<b> Sattisfacción laboral <b>',
             color_discrete_sequence=px.colors.qualitative.G10)

# agregar detalles a la gráfica
fig.update_layout(
    template = 'simple_white',
    legend_title = '<b> Labor <b>',
    title_x = 0.5)

fig.show()

#La mayoría de personas que renunciaron, habían votado por 3 en esta sección, el menor valor de votación fue el 2

"""*Algunas conclusiones de esta sección son: En 2/3 variables las personas que renunciaron en su mayoría voto un valor de 3, siendo así una calificación peor que la del valor 1.*"""

