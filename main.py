import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


st.set_page_config(
  page_title="Predicción de deserción de clientes",
  page_icon="icono.ico",
  initial_sidebar_state='auto',
  menu_items={
        'Report a bug': 'http://www.unab.edu.co',
        'Get Help': "https://docs.streamlit.io/get-started/fundamentals/main-concepts",
        'About': "Nathalia Quintero & Angelly Cristancho. Inteligencia Artificial *Ejemplo de clase* Ingenieria de sistemas!"
    }
  )

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introducción","Naive Bayes", "Modelo Arbol", "Modelo Bosque", "Modelo RL"])

with tab1:
    st.title("Hola! Bienvenido a mi pagina")
    with st.container( border=True):
        st.subheader("Modelo Machine Learning para predecir la deserción de clientes")
        #Se desea usar emoji lo puedes buscar aqui.
        st.write("""Realizado por Keren Nathalia Quintero &
                 Angely Gabriela Cristancho:\U0001F33B\U0001F42C:""")
        st.write("""

**Introducción: ** 
cliente de nuestros sueños es el que permanece fiel a la empresa, comprando siempre sus productos o servicios. Sin embargo, en la realidad, 
los clientes a veces deciden alejarse de la empresa para probar o empezar a comprar otros productos o servicios y esto puede ocurrir en 
cualquier fase del customer journey. Sin embargo, existen varias medidas para prevenir o gestionar mejor esta circunstancia. Por eso lo mejor
es tener una herramienta predictiva que nos indique el estado futuro de dichos clientes usando inteligencia artificial, tomar las acciones 
de retenció necesaria. Constituye pues esta aplicación una herramienta importante para la gestión del marketing.

Los datos fueron tomados con la Información de la base de datos CRM de la empresa ubicada en Bucaramanfa,donde se
preparó 3 modelos de machine Learnig para predecir la deserció de clientes, tanto actuales como nuevos.

Datos Actualizados en la fuente: 20 de Marzo del 2024


Se utilizó modelos supervidados de clasificacion  tanto Naive Bayes, Arboles de decisión y Bosques Aleatorios 
entendiendo que hay otras técnicas, es el resultado de la aplicacion practico del curso de inteligencia artificial en estos modelos
revisado en clase. Aunqe la aplicación final sería un solo modelo, aqui se muestran los tres modelos para 
comparar los resultados.

 """)
    with st.container(border=True,height=250):
        st.subheader("Detalles")
        st.write(""" Este es un ejemplo de despliegue de los modelos de Machine Learning entrenados en
                Google Colab con las librerias de scikit-learn par Naive Bayes, Arbles de Decisión y Bosques Aleatorios.
                En este notebook podrás verificar el preprocesamiento del dataset y el entrenamiento y las pruebas
                y scores obtenidos.
                https://colab.research.google.com/drive/1Lth_RqbnAnBVAMjSWinpXoTitI9OPaIv""")
        
with tab2:
    # Cargar el modelo de Naive Bayes
    modelo_nb = joblib.load('ModeloNb1.bin')
    # Cargar el modelo entrenado desde el archivo .pkl
    #with open('modelos/modelo_entrenadoNB.pkl', 'rb') as file:
       # modelo = pickle.load(file)


    st.title("Predicción con modelo de NB")
    st.write("Visualiza los datos ingresados aqui:")

    st.sidebar.write("<h1 style='font-weight:bold;'>INGRESE LOS DATOS</h1>", unsafe_allow_html=True)
    ANTIG_slider = st.sidebar.slider('ANTIGUEDAD',-0.0800, 15.0000)
    ANTIG = st.sidebar.number_input('', min_value=-0.0800, max_value=15.0000, value=ANTIG_slider)
    st.sidebar.write('---')  # Línea separadora
    COMP= st.sidebar.slider('COMPRAS', 4414, 18338)
    COMP_input = st.sidebar.number_input('', min_value=4414, max_value=18338, value=COMP)
    st.sidebar.write('---') 
    PROM= st.sidebar.slider('PROMEDIO DE COMPRAS', 0.85, 8.61)
    PROM_input = st.sidebar.number_input('', min_value=0.85, max_value=8.61, value=PROM)
    st.sidebar.write('---') 
    CATEG=st.sidebar.slider('COMPRAS POR CATEGORIA', 2206.99, 9168.78)
    CATEG_input = st.sidebar.number_input('', min_value=2206.99, max_value=9168.78, value=CATEG)
    st.sidebar.write('---') 
    COMINT=st.sidebar.slider('COMPRAS POR INTERNET', 1576, 51731)
    COMINT_input = st.sidebar.number_input('', min_value=1576, max_value=51731, value=COMINT)
    st.sidebar.write('---') 
    COMPPRES=st.sidebar.slider('COMPRAS PRESENCIALES', 19916, 90779)
    COMPPRES_input = st.sidebar.number_input('', min_value=19916, max_value=90779, value=COMPPRES)
    st.sidebar.write('---') 
    RATE=st.sidebar.slider('TASA DE COMPRAS (RATE)', 0.655972117, 4.16725588)
    RATE_input = st.sidebar.number_input('', min_value=0.655972117, max_value=4.16725588, value=RATE)
    st.sidebar.write('---') 
    VISIT=st.sidebar.slider('VISITAS', 2.0, 130.0)
    VISIT_input = st.sidebar.number_input('', min_value=2.0, max_value=130.0, value=VISIT)
    st.sidebar.write('---') 
    DIASSINQ=st.sidebar.slider('DIAS SIN Q', 370, 1785)
    DIASSINQ_input = st.sidebar.number_input('', min_value=370, max_value=1785, value=DIASSINQ)
    st.sidebar.write('---') 
    TASARET= st.sidebar.slider('RETENCIÓN', 0.399886065, 1.768134227)
    TASARET_input = st.sidebar.number_input('', min_value=0.399886065, max_value=1.768134227, value=TASARET)
    st.sidebar.write('---') 
    NUMQ=st.sidebar.slider('NUMERO DE COTIZACIONES', 3.207496745, 9.469748015)
    NUMQ_input = st.sidebar.number_input('', min_value=3.207496745, max_value=9.469748015, value=NUMQ)
    st.sidebar.write('---') 
    RETRE= st.sidebar.slider('TASA DE RETENCION', 3.348704033, 31.49416946)
    RETRE_input = st.sidebar.number_input('', min_value=3.348704033, max_value=31.49416946, value=RETRE)
    # Crear los datos para la tabla
    data = {
        'ANTIG': [ANTIG],
        'COMP': [COMP],
        'PROM': [PROM],
        'CATEG': [CATEG],
        'COMINT': [COMINT],
        'COMPPRES': [COMPPRES],
        'RATE': [RATE],
        'VISIT': [VISIT],
        'DIASSINQ': [DIASSINQ],
        'TASARET': [TASARET],
        'NUMQ': [NUMQ],
        'RETRE':[RETRE]
    }

    # Convertir los datos a un DataFrame de pandas
    df = pd.DataFrame(data)
    st.write(df)

    nueva_persona=[ANTIG,COMP,PROM,CATEG,COMINT,COMPPRES,RATE,VISIT,DIASSINQ,TASARET,NUMQ,RETRE]
    #nueva_persona=[-0.083457,12402,3.77,6200.96,28162,74018,2.233593,86.0,820,1.337147,7.737920,9.613742]#nuestro 0
    #nueva_persona=[-0.08345991631487736,6070,5.83,3035.04,19586,29311,1.913495922,58.0,1024,1.321100544,6.673170825,23.53717176]#nuestro 1
    #nueva_persona=[ANTIG,COMP,PROM,CATEG,COMINT,COMPPRES,1.913495922,58.0,1024,1.321100544,6.673170825,23.53717176]#nuestro 1
    #st.write(nueva_persona)
    nueva_persona_2d = np.array(nueva_persona).reshape(1, -1)

    # Mostrar la tabla en Streamlit
    #st.table(df)

    # Realizar predicciones con el modelo de Naive Bayes
    #prediccion_nb = modelo_nb.predict(df)

    # Mostrar la predicción
    #st.write("Predicción del modelo de Naive Bayes:", prediccion_nb)

    # Realizar la predicción con los modelos
    pred_NB = modelo_nb.predict(nueva_persona_2d)

    # Mostrar los resultados
    st.write("Predicción del modelo de Naive Bayes:", pred_NB[0])
    st.write("Usted es:")


    # Crear una tarjeta para mostrar el resultado
    result_card = st.empty()
    # Función para mostrar el resultado con animación
    def show_result(is_positive):
        if pred_NB[0]==1:
            # Tarjeta verde con mensaje de "POSITIVO"
            result_card.error("**El CLIENTE VA A DESERTAR**")
        else:
            # Tarjeta roja con mensaje de "NEGATIVO"
            result_card.success("**El CLIENTE NO VA A DESERTAR**")

    # Llamar a la función para mostrar el resultado inicialmente
    show_result(True)  # Cambia a False si el resultado es NEGATIVO

    probabilidad_NB=modelo_nb.predict_proba(nueva_persona_2d)
    col1, col2= st.columns(2)
    col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad_NB[0][0]),delta=" ")
    col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad_NB[0][1]),delta=" ")
    
    


with tab3:
    #Cargar el modelo de Arbol
    modelo_arbol = joblib.load('ModeloArbol.bin')
    
    st.title("Predicción con modelo Arbol")
    st.write("Visualiza los datos ingresados aqui:")

    # Crear los datos para la tabla
    data = {
        'ANTIG': [ANTIG],
        'COMP': [COMP],
        'PROM': [PROM],
        'CATEG': [CATEG],
        'COMINT': [COMINT],
        'COMPPRES': [COMPPRES],
        'RATE': [RATE],
        'VISIT': [VISIT],
        'DIASSINQ': [DIASSINQ],
        'TASARET': [TASARET],
        'NUMQ': [NUMQ],
        'RETRE':[RETRE]
    }

    # Convertir los datos a un DataFrame de pandas
    df = pd.DataFrame(data)
    st.write(df)

    nueva_persona=[ANTIG,COMP,PROM,CATEG,COMINT,COMPPRES,RATE,VISIT,DIASSINQ,TASARET,NUMQ,RETRE]
    nueva_persona_2d = np.array(nueva_persona).reshape(1, -1)

    # Realizar la predicción con los modelos
    pred_Arbol = modelo_arbol.predict(nueva_persona_2d)

    # Mostrar los resultados
    st.write("Predicción del modelo de Arbol:", pred_Arbol[0])
    st.write("Usted es:")


    # Crear una tarjeta para mostrar el resultado
    result_card = st.empty()
    # Función para mostrar el resultado con animación
    def show_result(is_positive):
        if pred_Arbol[0]==1:
            # Tarjeta verde con mensaje de "POSITIVO"
            result_card.error("**El CLIENTE VA A DESERTAR**")
        else:
            # Tarjeta roja con mensaje de "NEGATIVO"
            result_card.success("**El CLIENTE NO VA A DESERTAR**")

    # Llamar a la función para mostrar el resultado inicialmente
    show_result(True)  # Cambia a False si el resultado es NEGATIVO

    probabilidad_arbol=modelo_arbol.predict_proba(nueva_persona_2d)
    importancia_arbol=modelo_arbol.feature_importances_
    col1, col2= st.columns(2)
    col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad_arbol[0][0]),delta=" ")
    col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad_arbol[0][1]),delta=" ")
    features=modelo_arbol.feature_names_in_
    importancia_arbol=pd.Series(importancia_arbol,index=features)
    st.bar_chart(importancia_arbol)

   
with tab4:
    # Cargar el modelo de Naive Bayes
    modelo_bosque = joblib.load('modelobosque.bin')

    st.title("Predicción con modelo Bosque")
    st.write("Visualiza los datos ingresados aqui:")

    # Crear los datos para la tabla
    data = {
        'ANTIG': [ANTIG],
        'COMP': [COMP],
        'PROM': [PROM],
        'CATEG': [CATEG],
        'COMINT': [COMINT],
        'COMPPRES': [COMPPRES],
        'RATE': [RATE],
        'VISIT': [VISIT],
        'DIASSINQ': [DIASSINQ],
        'TASARET': [TASARET],
        'NUMQ': [NUMQ],
        'RETRE':[RETRE]
    }

    # Convertir los datos a un DataFrame de pandas
    df = pd.DataFrame(data)
    st.write(df)

    nueva_persona=[ANTIG,COMP,PROM,CATEG,COMINT,COMPPRES,RATE,VISIT,DIASSINQ,TASARET,NUMQ,RETRE]
    nueva_persona_2d = np.array(nueva_persona).reshape(1, -1)

    # Realizar la predicción con los modelos
    pred_bosque = modelo_bosque.predict(nueva_persona_2d)

    # Mostrar los resultados
    st.write("Predicción del modelo de Arbol:", pred_bosque[0])
    st.write("Usted es:")


    # Crear una tarjeta para mostrar el resultado
    result_card = st.empty()
    # Función para mostrar el resultado con animación
    def show_result(is_positive):
        if pred_bosque[0]==1:
            # Tarjeta verde con mensaje de "POSITIVO"
            result_card.error("**El CLIENTE VA A DESERTAR**")
        else:
            # Tarjeta roja con mensaje de "NEGATIVO"
            result_card.success("**El CLIENTE NO VA A DESERTAR**")

    # Llamar a la función para mostrar el resultado inicialmente
    show_result(True)  # Cambia a False si el resultado es NEGATIVO

    probabilidad_bosque=modelo_bosque.predict_proba(nueva_persona_2d)
    importancia_bosque=modelo_bosque.feature_importances_
    col1, col2= st.columns(2)
    col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad_bosque[0][0]),delta=" ")
    col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad_bosque[0][1]),delta=" ")
    features=modelo_bosque.feature_names_in_
    importancia_bosque=pd.Series(importancia_arbol,index=features)
    st.bar_chart(importancia_bosque)

with tab5:
     #Cargar el modelo de Arbol
    modelo_rl = joblib.load('ModeloRL.bin')
    
    st.title("Predicción con modelo RL")
    st.write("Visualiza los datos ingresados aqui:")

    # Crear los datos para la tabla
    data = {
        'ANTIG': [ANTIG],
        'COMP': [COMP],
        'PROM': [PROM],
        'CATEG': [CATEG],
        'COMINT': [COMINT],
        'COMPPRES': [COMPPRES],
        'RATE': [RATE],
        'VISIT': [VISIT],
        'DIASSINQ': [DIASSINQ],
        'TASARET': [TASARET],
        'NUMQ': [NUMQ],
        'RETRE':[RETRE]
    }

    # Convertir los datos a un DataFrame de pandas
    df = pd.DataFrame(data)
    st.write(df)

    nueva_persona=[ANTIG,COMP,PROM,CATEG,COMINT,COMPPRES,RATE,VISIT,DIASSINQ,TASARET,NUMQ,RETRE]
    nueva_persona_2d = np.array(nueva_persona).reshape(1, -1)

    # Realizar la predicción con los modelos
    pred_rl = modelo_rl.predict(nueva_persona_2d)

    # Mostrar los resultados
    st.write("Predicción del modelo RL:", pred_rl[0])
    st.write("Usted es:")


    # Crear una tarjeta para mostrar el resultado
    result_card = st.empty()
    # Función para mostrar el resultado con animación
    def show_result(is_positive):
        if pred_rl[0]==1:
            # Tarjeta verde con mensaje de "POSITIVO"
            result_card.error("**El CLIENTE VA A DESERTAR**")
        else:
            # Tarjeta roja con mensaje de "NEGATIVO"
            result_card.success("**El CLIENTE NO VA A DESERTAR**")

    # Llamar a la función para mostrar el resultado inicialmente
    show_result(True)  # Cambia a False si el resultado es NEGATIVO

    probabilidad_rl=modelo_rl.predict_proba(nueva_persona_2d)
    importancia_rl=modelo_rl.feature_importances_
    col1, col2= st.columns(2)
    col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad_rl[0][0]),delta=" ")
    col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad_rl[0][1]),delta=" ")
    #features=modelo_rl.feature_names_in_
    #importancia_rl=pd.Series(importancia_rl,index=features)
    #st.bar_chart(importancia_rl)
