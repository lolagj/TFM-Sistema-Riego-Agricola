import streamlit as st
import mlflow
import os
import altair as alt
import plotly.express as px
import pandas as pd
from datetime import date
from joblib import load
from src.transform_input import coordenadas_gms, transform_dia_año, transform_mes_año, extract_fecha
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(
    page_title="Sistema de ayuda a la toma de desición Riego Agrícola: Olivo",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded")

alt.theme.enable("dark")

df = pd.read_csv("data/demo_app/df_demo.csv", sep =";")

coordenadas = df[['nombre','provincia_nombre','xutm', 'yutm', 'altitud']].drop_duplicates()
lon, lat =coordenadas_gms(coordenadas["xutm"],coordenadas["yutm"])

with st.sidebar:
    st.title('💧 Sistema de ayuda a la toma de decisión de Riego Agrícola: Olivo')
    
    mun_list = sorted(list(df['nombre'].unique()), reverse=False)
    selected_mun = st.selectbox('Selecciona un municipio', mun_list, index=len(mun_list)-1)
    df_selected_mun = df[df.nombre == selected_mun]
    cultivo = st.selectbox('Seleccione cultivo', "Olivo")

    df['fecha'] = pd.to_datetime(df['fecha'])

    if selected_mun:
        df_mun = df[df['nombre']==selected_mun]
        fecha_min = df_mun['fecha'].min()
        fecha_max = df_mun['fecha'].max()
        st.write(f"Datos disponibles para esta estación desde el {fecha_min} hasta el {fecha_max}")


hoy = date.today()
año, mes, dia_año = extract_fecha(hoy)
sin_mes, cos_mes = transform_mes_año(mes)
sin_dia_año, cos_dia_año = transform_dia_año(dia_año)

modelo = load("notebooks/modelado/XGBoost_modelo_final.joblib")

scaler = load('notebooks/preparacion_datos/standard_scaler.joblib')


st.header(f"Predicción de Evapotranspiración para hoy ({hoy})")
with st.form("formulario_prediccion"):
    st.write("Introduce los parámetros meteorológicos diarios:")
    col1, col2, col3= st.columns(3)
    with col1:
        temp_max = st.number_input(
            "Temperatura Máxima (°C)", 
            min_value=0.0, 
            max_value=50.0, 
            value=22.5, 
            step=0.1
        )
        temp_min = st.number_input(
            "Temperatura Mínima (°C)", 
            min_value=-20.0, 
            max_value=40.0, 
            value=15.0, 
            step=0.1
        )
    with col2:
        humedad_max = st.slider(
            "Humedad Relativa Máxima(%)", 
            min_value=0, 
            max_value=100, 
            value=65
        )   
        humedad_min = st.slider(
            "Humedad Relativa Mínima (%)", 
            min_value=0, 
            max_value=100, 
               value=30
        )          
    with col3:
        viento = st.number_input(
            "Velocidad del Viento (m/s)", 
            min_value=0.0, 
            max_value=20.0, 
            value=2.0, 
            step=0.1
        )
        submitted = st.form_submit_button("Ejecutar Predicción")
mun_data = coordenadas[coordenadas['nombre']==selected_mun]
lon, lat = coordenadas_gms(mun_data.xutm, mun_data.yutm)
if submitted:
    st.markdown("---")
    st.subheader("Resultado de la Predicción")
    datos_para_modelo = {
        'tempMax': temp_max,
        'tempMin': temp_min,
        'humedadMax': humedad_max,
        'humedadMin': humedad_min,
        'velViento': viento,
        'altitud': mun_data.altitud.iloc[0],
        'lon': lon,
        'lat': lat,
        'dia_del_año_sin': sin_dia_año,
        'dia_del_año_cos': cos_dia_año, 
        'año': año, 
        'mes': mes, 
        'mes_sin': sin_mes, 
        'mes_cos': cos_mes
    }
    #Scaler entrenado con conjunto completo. Como estamos evaluando subconjunto, es necesario rehacerlo.
    columnas_originales = [
    'tempMedia', 'tempMax', 'tempMin', 'humedadMedia', 'humedadMax',
    'humedadMin', 'velViento', 'dirViento', 'velVientoMax',
    'dirVientoVelMax', 'radiacion', 'precipitacion', 'altitud', 'lon',
    'lat', 'dia_del_año_sin', 'dia_del_año_cos', 'año', 'mes', 'mes_sin',
    'mes_cos']
    columnas_subset = [
    'tempMax', 'tempMin', 'humedadMax', 'humedadMin', 'velViento', 
    'altitud', 'lon', 'lat', 'dia_del_año_sin', 'dia_del_año_cos', 
    'año', 'mes', 'mes_sin', 'mes_cos']
    indices_para_conservar = [columnas_originales.index(col) for col in columnas_subset]
    media_original = scaler.mean_
    escala_original = scaler.scale_  
    media_subset = media_original[indices_para_conservar]
    escala_subset = escala_original[indices_para_conservar]
    scaler_subset = StandardScaler()
    scaler_subset.mean_ = media_subset
    scaler_subset.scale_ = escala_subset
    scaler_subset.n_features_in_ = len(columnas_subset)
    input = pd.DataFrame(datos_para_modelo, index=[0])
    input = scaler_subset.transform(input.astype(float))
    print(input)
    with st.spinner("Ejecutando modelo..."):
        resultado_prediccion = modelo.predict(input)
    st.success("¡Predicción completada!")
    col3, col4= st.columns(2)
    with col3:
        st.metric(
            label="Evapotranspiración de referencia (ETo) resultante",
            value=f"{resultado_prediccion[0]:.2f} mm/día"
        )
    kc_mes = {
        12: 0.5,
        1: 0.5,
        2: 0.5,
        3: 0.65,
        4: 0.65,
        5: 0.65,
        6: 0.60,
        7: 0.60,
        8: 0.60,
        9: 0.60,
        10:0.65,
        11:0.65
    }
    with col4:
        st.metric(
            label=f"Evapotranspiración del cultivo (ETc) calculada, tomando como Kc: {kc_mes.get(mes)}",
            value=f"{resultado_prediccion[0]*kc_mes.get(mes):.2f} mm/día"
        )    
    
 