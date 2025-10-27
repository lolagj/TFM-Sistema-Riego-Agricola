from pyproj import Transformer 
import numpy as np
import folium
def coordenadas_gms(x,y):
    transformer = Transformer.from_crs("epsg:32630", "epsg:4326", always_xy=True)       
    lon, lat = transformer.transform(x, y)
    return lon, lat

def transform_dia_año(x):
    sin = np.sin(2 * np.pi * x/365)
    cos = np.cos(2 * np.pi *x/365)
    return sin, cos

def transform_mes_año(x):
    sin= np.sin(2 * np.pi * x/12)
    cos = np.cos(2 * np.pi * x/12)
    return sin, cos

def extract_fecha(x):
    año = x.year
    mes = x.month
    dia_año = x.timetuple().tm_yday
    return año, mes, dia_año
