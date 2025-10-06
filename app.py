import os
import json
from datetime import datetime
import math
import pickle
import joblib
import numpy as np
from scipy.interpolate import griddata
import requests
from flask import Flask, render_template, request, jsonify, url_for, redirect, session, send_from_directory
import mysql.connector
import bcrypt

# ---------------- CONFIGURACIÓN ----------------
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'mi_clave_secreta_super_segura')

MODEL_PATH = "datos_climaticos_completos.pkl"
URL = "https://www.dropbox.com/scl/fi/8v7yr1v1vsb5sktoz1dcs/datos_climaticos_completos.pkl?rlkey=00zqgjwnuah9tsyxjtkkzlds5&dl=1"

# Descarga el modelo si no existe localmente
if not os.path.exists(MODEL_PATH):
    print("📦 Descargando modelo desde Dropbox...")
    r = requests.get(URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):  # descarga en bloques
            f.write(chunk)
    print("✅ Modelo descargado correctamente.")
else:
    print("📂 Modelo ya existe localmente.")

# ---------------- FUNCIONES CLIMÁTICAS ----------------

def cargar_parcial(fecha_hora):
    """
    Carga solo la parte del modelo necesaria para una fecha/hora específica.
    Evita cargar todo el .pkl (reduce uso de RAM).
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            while True:
                try:
                    bloque = pickle.load(f)
                    if fecha_hora in bloque:
                        return bloque[fecha_hora]
                except EOFError:
                    break
    except Exception as e:
        print(f"❌ Error cargando modelo parcial: {e}")
    return None

def calcular_humedad_relativa(temp_c, pto_rocio_c):
    if temp_c is None or pto_rocio_c is None:
        return None
    b = 17.625
    c = 243.04
    gamma = (b * pto_rocio_c / (c + pto_rocio_c)) - (b * temp_c / (c + temp_c))
    rh = 100 * math.exp(gamma)
    return min(100, max(0, rh))

def pronosticar_clima(latitud: float, longitud: float, fecha: str, hora: str):
    """
    Pronostica usando carga parcial del modelo.
    """
    if not os.path.exists(MODEL_PATH):
        return {"error": "Modelo no disponible."}

    if len(hora.split(':')) == 1:
        hora += ":00"
    fecha_hora_str = f"{fecha} {hora}"

    try:
        fecha_obj = datetime.strptime(fecha_hora_str, '%Y-%m-%d %H:%M')
        mes_dia_hora = fecha_obj.strftime('%m-%d %H:00:00')
        anio_futuro = fecha_obj.year
    except (ValueError, TypeError):
        return {"error": f"Formato de fecha/hora inválido: {fecha_hora_str}"}

    resultados = {}
    for variable in ['temperatura', 'humedad', 'precipitacion']:
        valores_historicos, anios_historicos = [], []
        for anio in range(2015, 2025):
            fecha_historica_str = f"{anio}-{mes_dia_hora}"
            bloque = cargar_parcial(fecha_historica_str)
            if bloque and 'puntos' in bloque and variable in bloque:
                valor_estimado = griddata(bloque['puntos'], bloque[variable], (longitud, latitud), method='cubic')
                if not np.isnan(valor_estimado):
                    valores_historicos.append(float(valor_estimado))
                    anios_historicos.append(anio)
        if len(valores_historicos) < 4:
            resultado_final = np.mean(valores_historicos) if valores_historicos else None
        else:
            pendiente, intercepto = np.polyfit(anios_historicos, valores_historicos, 1)
            resultado_final = (pendiente * anio_futuro) + intercepto
        resultados[variable] = resultado_final
    return resultados

def generar_descripcion_completa(resultados):
    temp = resultados.get('temperatura')
    precip = resultados.get('precipitacion')
    if temp is None: desc_temp = "Temperatura no disponible"
    elif temp < 5: desc_temp = "Muy Frío"
    elif 5 <= temp < 12: desc_temp = "Frío"
    elif 12 <= temp < 18: desc_temp = "Fresco / Templado"
    elif 18 <= temp < 24: desc_temp = "Cálido / Agradable"
    else: desc_temp = "Caluroso"
    if precip is None or precip < 0:
        desc_precip = ""
    elif precip <= 0.1: desc_precip = "con cielo mayormente despejado."
    elif 0.1 < precip <= 1.0: desc_precip = "con posibles lloviznas."
    elif 1.0 < precip <= 5.0: desc_precip = "con probabilidad de lluvia."
    else: desc_precip = "con lluvias intensas."
    temp_str = f"{temp:.1f}°C" if isinstance(temp, float) else "N/A"
    return f"El pronóstico es {desc_temp} (aprox. {temp_str}) {desc_precip}"

def generar_descripcion_corta(resultados):
    temp = resultados.get('temperatura')
    precip = resultados.get('precipitacion')
    if precip is not None and precip > 1.0: return "Lluvioso"
    if temp is None: return "Desconocido"
    if temp > 24: return "Caluroso"
    if temp > 18: return "Cálido"
    if temp > 12: return "Templado"
    return "Frío"

def obtener_ubicacion_osm(latitud, longitud):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitud}&lon={longitud}"
        headers = {'User-Agent': 'EcoWeatherApp/1.0'}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        if 'address' in data:
            addr = data['address']
            ciudad = addr.get('city', addr.get('town', addr.get('village', addr.get('state'))))
            return ciudad or 'Ubicación desconocida', addr.get('country', 'N/A')
        return "Sin datos", "N/A"
    except Exception:
        return "Error de API", "N/A"

# ---------------- BASE DE DATOS ----------------
def conectar(con_db=True):
    cfg = {"host": "localhost", "user": "root", "password": ""}
    if con_db:
        cfg["database"] = "login_db"
    return mysql.connector.connect(**cfg)

def inicializar_db():
    try:
        conn = conectar(con_db=False)
        c = conn.cursor()
        c.execute("CREATE DATABASE IF NOT EXISTS login_db")
        conn.close()
        conn = conectar(con_db=True)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS eventos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                usuario_id INT NOT NULL,
                titulo VARCHAR(255) NOT NULL,
                descripcion TEXT,
                fecha_evento DATETIME NOT NULL,
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
            )
        """)
        conn.commit()
        print("✅ DB inicializada correctamente.")
    except Exception as e:
        print(f"❌ Error DB: {e}")

# ---------------- RUTAS PRINCIPALES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_climate_data', methods=['POST'])
def get_climate_data():
    data = request.json
    lat, lon, date, time = data.get('latitude'), data.get('longitude'), data.get('date'), data.get('time')
    pred = pronosticar_clima(lat, lon, date, time)
    desc = generar_descripcion_completa(pred)
    ciudad, pais = obtener_ubicacion_osm(lat, lon)
    return jsonify({'departamento': ciudad, 'pais': pais, 'descripcion': desc})

# --- y todas tus demás rutas (login, registro, agenda, chatbot, etc.) siguen IGUALES ---
# (no las elimino para no romper el proyecto, pero no las repito aquí por espacio)

if __name__ == '__main__':
    inicializar_db()
    app.run(debug=True, port=8000)
