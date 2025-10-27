import streamlit as st
import mlflow
import os
import altair as alt
import plotly.express as px
import pandas as pd
from datetime import date
from joblib import load

st.set_page_config(
    page_title="Sistema de ayuda a la toma de desición Riego Agrícola: Olivo",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded")

alt.theme.enable("dark")

df = pd.read_csv("data/demo_app/df_demo.csv", sep =";")


with st.sidebar:
    st.title('💧 Sistema de ayuda a la toma de desición Riego Agrícola: Olivo')
    
    mun_list = list(df['nombre'].unique())[::-1]
    selected_mun = st.selectbox('Selecciona un municipio', mun_list, index=len(mun_list)-1)
    df_selected_mun = df[df.nombre == selected_mun]
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    cultivo = st.selectbox('Seleccione cultivo', "Olivo")

    df['fecha'] = pd.to_datetime(df['fecha'])

    if selected_mun:
        df_mun = df[df['nombre']==selected_mun]
        fecha_min = df_mun['fecha'].min()
        fecha_max = df_mun['fecha'].max()
        st.write(f"Datos disponibles para esta estación desde el {fecha_min} hasta el {fecha_max}")
        rango_seleccionado = st.date_input("Selecciona un rango de fechas",
            value=(fecha_min, fecha_max), 
            min_value=fecha_min,               
            max_value=fecha_max               
        )

        if len(rango_seleccionado) == 2:
            fecha_inicio = rango_seleccionado[0]
            fecha_fin = rango_seleccionado[1]
            
            st.success(f"Rango seleccionado: {fecha_inicio} al {fecha_fin}")
            
            
        else:
            st.warning("No hay datos para la estación seleccionada.")


hoy = date.today()



modelo = load('notebooks/modelado/XGBoost_modelo_final.joblib')


#model.predict()

