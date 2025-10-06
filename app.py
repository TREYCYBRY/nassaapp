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

# El modelo se descargará solo cuando sea necesario
modelo_disponible = False


def verificar_o_descargar_modelo():
    """
    Descarga el modelo solo si no existe localmente.
    Se ejecuta una sola vez antes de usarlo.
    """
    global modelo_disponible
    if modelo_disponible:
        return True

    if not os.path.exists(MODEL_PATH):
        print("📦 Descargando modelo desde Dropbox (modo diferido)...")
        try:
            with requests.get(URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print("✅ Modelo descargado correctamente.")
            modelo_disponible = True
        except Exception as e:
            print(f"❌ Error al descargar modelo: {e}")
            return False
    else:
        modelo_disponible = True
    return True


# ---------------- FUNCIONES CLIMÁTICAS ----------------
def cargar_parcial(fecha_hora):
    """
    Carga solo la parte del modelo necesaria para una fecha/hora específica.
    Evita cargar todo el .pkl (reduce uso de RAM).
    """
    if not verificar_o_descargar_modelo():
        return None

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
    """Calcula la humedad relativa en % a partir de la temperatura y el punto de rocío."""
    if temp_c is None or pto_rocio_c is None:
        return None
    b = 17.625
    c = 243.04
    gamma = (b * pto_rocio_c / (c + pto_rocio_c)) - (b * temp_c / (c + temp_c))
    rh = 100 * math.exp(gamma)
    return min(100, max(0, rh)) # Asegura que esté entre 0 y 100

def pronosticar_clima(latitud: float, longitud: float, fecha: str, hora: str):
    """
    Pronostica usando carga parcial del modelo.
    """
    if not verificar_o_descargar_modelo():
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

    if temp is None:
        desc_temp = "Temperatura no disponible"
    elif temp < 5:
        desc_temp = "Muy Frío"
    elif 5 <= temp < 12:
        desc_temp = "Frío"
    elif 12 <= temp < 18:
        desc_temp = "Fresco / Templado"
    elif 18 <= temp < 24:
        desc_temp = "Cálido / Agradable"
    else:
        desc_temp = "Caluroso"

    if precip is None or precip < 0:
        desc_precip = ""
    elif precip <= 0.1:
        desc_precip = "con cielo mayormente despejado."
    elif 0.1 < precip <= 1.0:
        desc_precip = "con posibles lloviznas."
    elif 1.0 < precip <= 5.0:
        desc_precip = "con probabilidad de lluvia."
    else:
        desc_precip = "con lluvias intensas."

    temp_str = f"{temp:.1f}°C" if isinstance(temp, float) else "N/A"
    return f"El pronóstico es {desc_temp} (aprox. {temp_str}) {desc_precip}"


def generar_descripcion_corta(resultados):
    temp = resultados.get('temperatura')
    precip = resultados.get('precipitacion')
    if precip is not None and precip > 1.0:
        return "Lluvioso"
    if temp is None:
        return "Desconocido"
    if temp > 24:
        return "Caluroso"
    if temp > 18:
        return "Cálido"
    if temp > 12:
        return "Templado"
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

@app.route('/inicio')
def inicio():
    return render_template('index.html')

@app.route('/agenda')
def agenda():
    return render_template('agenda.html')

@app.route('/info')
def info_clima():
    return render_template('info_clima.html')

@app.route('/pronostico')
def pronostico():
    return render_template('pronost.html') # La página principal del clima

@app.route('/login')
def iniciosesion():
    return render_template('login/login.html')

@app.route('/registro')
def registrousu():
    return render_template('login/registro.html')

@app.route('/juego')
def juego():
    return send_from_directory('juego', 'inicio.html')

@app.route('/juego/clasico')
def clasico():
    return send_from_directory('juego', 'clasico.html')

@app.route('/juego/rompecabezas_clasico')
def rompecabezas_clasico():
    return send_from_directory('juego', 'rompecabezas_clasico.html')

@app.route('/juego/personalizado')
def personalizado():
    return send_from_directory('juego', 'personalizado.html')


@app.route('/juego/rompecabeza')
def rompecabeza():
    return send_from_directory('juego', 'rompecabeza.html')

@app.route('/educacion')
def educacion():
    return render_template('educacion/educacion.html')

# --- ¡NUEVA RUTA AÑADIDA! Para la página de comparación ---
@app.route('/comparar')
def comparar():
    return render_template('comparar.html')

@app.route('/perfil')
def perfil():
    if 'usuario' in session:
        # Si está logueado, mostrar su información
        return render_template('login/profile.html', usuario=session['usuario'])
    else:
        # Si NO está logueado, mostrar opciones
        return render_template('login/perfil_opciones.html')

# --- API para la funcionalidad del clima (REFORMULADA Y MEJORADA) ---
@app.route('/api/get_climate_data', methods=['POST'])
def get_climate_data():
    data = request.json
    try:
        lat, lon, date, time = data.get('latitude'), data.get('longitude'), data.get('date'), data.get('time')
    except Exception:
        return jsonify({'error': 'Faltan datos en la solicitud.'}), 400
    
    # 1. Obtener el pronóstico numérico con las 3 variables
    prediccion_numerica = pronosticar_clima(lat, lon, date, time)
    
    # 2. Generar la descripción en texto
    descripcion = generar_descripcion_completa(prediccion_numerica)
    
    # 3. Obtener el nombre del lugar
    departamento, pais = obtener_ubicacion_osm(lat, lon)
    
    return jsonify({
        'departamento': departamento,
        'pais': pais,
        'descripcion': descripcion
    })

# --- API para el gráfico diario (FUNCIONALIDAD AÑADIDA) ---
@app.route('/api/daily_chart', methods=['POST'])
def daily_chart():
    data = request.json
    lat, lon, date = data.get('latitude'), data.get('longitude'), data.get('date')

    horas = [f"{h:02d}:00" for h in range(24)]
    temperaturas, puntos_rocio, precipitaciones = [], [], []

    for hora in horas:
        resultado = pronosticar_clima(lat, lon, date, hora)
        temp = resultado.get('temperatura')
        hum = resultado.get('humedad')  # Punto de rocío o humedad
        prec = resultado.get('precipitacion')

        temperaturas.append(temp if temp is not None else None)
        puntos_rocio.append(hum if hum is not None else None)
        precipitaciones.append(prec if prec is not None else 0)

    return jsonify({
        'labels': horas,
        'temperatures': temperaturas,
        'humidities': puntos_rocio,
        'precipitations': precipitaciones
    })

# --- ¡NUEVA API AÑADIDA! Para la página de comparación ---
@app.route('/api/get_comparison_data', methods=['POST'])
def get_comparison_data():
    data = request.json
    try:
        lat, lon, date, time = data.get('latitude'), data.get('longitude'), data.get('date'), data.get('time')
    except Exception:
        return jsonify({'error': 'Faltan datos en la solicitud.'}), 400
    
    prediccion = pronosticar_clima(lat, lon, date, time)
    humedad_relativa = calcular_humedad_relativa(prediccion.get('temperatura'), prediccion.get('humedad'))
    descripcion = generar_descripcion_corta(prediccion)
    ciudad, pais = obtener_ubicacion_osm(lat, lon)
    
    # Formatear la precipitación
    precip_val = prediccion.get('precipitacion')
    precipitacion_str = f"{precip_val:.1f}" if precip_val is not None and precip_val > 0 else "0"

    return jsonify({
        'ciudad': ciudad,
        'pais': pais,
        'fecha': date,
        'hora': time,
        'pronostico': descripcion,
        'temp': f"{prediccion.get('temperatura'):.1f}" if prediccion.get('temperatura') is not None else "N/A",
        'precipitacion': precipitacion_str,
        'humedad': f"{humedad_relativa:.0f}" if humedad_relativa is not None else "N/A"
    })

@app.route('/api/search_location', methods=['POST'])
def search_location():
    data = request.json
    place_name = data.get('place_name')
    if not place_name:
        return jsonify({'error': 'No se proporcionó un nombre de lugar.'}), 400
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={place_name}, Peru&format=json&limit=1"
        headers = {'User-Agent': 'MiAppClima/1.0'}
        respuesta = requests.get(url, headers=headers, timeout=10)
        datos = respuesta.json()
        if datos:
            result = datos[0]
            return jsonify({ "place_name": result.get("display_name"), "latitude": float(result.get("lat")), "longitude": float(result.get("lon")) })
        else:
            return jsonify({'error': f"No se encontraron resultados para '{place_name}'."}), 404
    except Exception as e:
        return jsonify({'error': f"Error en la API de búsqueda: {e}"}), 500
# --- ¡NUEVAS APIS PARA LA AGENDA! ---
@app.route('/api/agendar_evento', methods=['POST'])
def agendar_evento():
    if 'usuario' not in session:
        return jsonify({'error': 'No autorizado', 'reason': 'No has iniciado sesión.'}), 401
    
    data = request.json
    titulo = data.get('titulo')
    fecha = data.get('fecha')
    hora = data.get('hora')
    descripcion = data.get('descripcion', '')

    if not all([titulo, fecha, hora]):
        return jsonify({'error': 'Faltan datos en la solicitud.'}), 400

    fecha_evento_str = f"{fecha} {hora}"
    
    conexion = None
    try:
        conexion = conectar()
        cursor = conexion.cursor(dictionary=True)
        
        cursor.execute("SELECT id FROM usuarios WHERE username = %s", (session['usuario'],))
        user = cursor.fetchone()
        if not user:
            return jsonify({'error': 'Usuario no encontrado en la base de datos.'}), 404
        usuario_id = user['id']

        insert_cursor = conexion.cursor()
        sql_query = "INSERT INTO eventos (usuario_id, titulo, descripcion, fecha_evento) VALUES (%s, %s, %s, %s)"
        # --- ¡CORRECCIÓN! Se ha eliminado el 'usuario_id' duplicado ---
        sql_values = (usuario_id, titulo, descripcion, fecha_evento_str)
        insert_cursor.execute(sql_query, sql_values)
        
        conexion.commit()
        insert_cursor.close()
        return jsonify({'mensaje': 'Evento agendado con éxito'}), 201
    except Exception as e:
        if conexion: conexion.rollback()
        print(f"❌ Error al agendar evento en la base de datos: {e}")
        return jsonify({'error': f'Error en el servidor: {e}'}), 500
    finally:
        if conexion:
            conexion.close()

@app.route('/api/get_events', methods=['GET'])
def get_events():
    if 'usuario' not in session:
        return jsonify({'error': 'No autorizado'}), 401

    conexion = None
    try:
        conexion = conectar()
        cursor = conexion.cursor(dictionary=True)
        # Obtenemos el ID del usuario actual
        cursor.execute("SELECT id FROM usuarios WHERE username = %s", (session['usuario'],))
        user = cursor.fetchone()
        if not user:
            return jsonify({'error': 'Usuario no encontrado'}), 404
        user_id = user['id']

        # Buscamos todos los eventos de ese usuario
        cursor.execute("SELECT lugar, fecha_hora, descripcion FROM eventos WHERE user_id = %s", (user_id,))
        eventos_db = cursor.fetchall()
        
        # Formateamos los eventos para el frontend
        eventos_json = {}
        for evento in eventos_db:
            fecha = evento['fecha_hora'].strftime('%Y-%m-%d')
            if fecha not in eventos_json:
                eventos_json[fecha] = []
            eventos_json[fecha].append({
                'title': evento['lugar'],
                'desc': evento['descripcion']
            })
        
        cursor.close()
        return jsonify(eventos_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conexion:
            conexion.close()

@app.route('/api/obtener_eventos', methods=['GET'])
def obtener_eventos():
    if 'usuario' not in session:
        return jsonify({'error': 'No autorizado'}), 401

    conexion = None
    try:
        conexion = conectar()
        cursor = conexion.cursor(dictionary=True)
        
        # 1. Obtener el ID del usuario actual de la sesión
        cursor.execute("SELECT id FROM usuarios WHERE username = %s", (session['usuario'],))
        user = cursor.fetchone()
        if not user:
            return jsonify({'error': 'Usuario no encontrado'}), 404
        usuario_id = user['id']

        # 2. Buscar todos los eventos de ese usuario en la base de datos
        cursor.execute("SELECT titulo, descripcion, fecha_evento FROM eventos WHERE usuario_id = %s", (usuario_id,))
        eventos_db = cursor.fetchall()
        
        # 3. Formatear los eventos al formato JSON que el calendario espera
        eventos_para_frontend = {}
        for evento in eventos_db:
            fecha_str = evento['fecha_evento'].strftime('%Y-%m-%d')
            if fecha_str not in eventos_para_frontend:
                eventos_para_frontend[fecha_str] = []
            
            eventos_para_frontend[fecha_str].append({
                'title': evento['titulo'],
                'desc': evento['descripcion']
            })
        
        cursor.close()
        return jsonify(eventos_para_frontend)

    except Exception as e:
        print(f"❌ Error al obtener eventos: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if conexion:
            conexion.close()
# --- Rutas de Autenticación de Usuarios (CORREGIDAS Y UNIFICADAS) ---
@app.route('/login', methods=['POST'])
def login():
    usuario = request.form.get('username')
    contraseña = request.form.get('password')
    
    if not usuario or not contraseña:
        return render_template('login/login.html', mensaje="❌ Complete usuario y contraseña.")

    conexion = None
    mensaje = ""
    try:
        conexion = conectar()
        cursor = conexion.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE username = %s", (usuario,))
        usuario_db = cursor.fetchone()
        cursor.close()

        if usuario_db and 'password' in usuario_db:
            stored = usuario_db['password']
            stored_bytes = stored.encode('utf-8') if isinstance(stored, str) else stored

            if bcrypt.checkpw(contraseña.encode('utf-8'), stored_bytes):
                # --- Guardamos usuario en sesión ---
                session['usuario'] = usuario
                mensaje = f"✅ Bienvenido, {usuario}"
                # Redirigir al perfil si quieres que inicie sesión automáticamente
                return redirect(url_for('inicio'))
            else:
                mensaje = "❌ Usuario o contraseña incorrectos."
        else:
            mensaje = "❌ Usuario o contraseña incorrectos."

    except Exception as e:
        mensaje = f"⚠️ Error en la conexión: {e}"
    finally:
        if conexion:
            conexion.close()

    # Si hubo error, mostramos login con mensaje
    return render_template('login/login.html', mensaje=mensaje)



@app.route('/registrar', methods=['POST'])
def registrar():
    usuario = request.form.get('username')
    contraseña = request.form.get('password')
    if not usuario or not contraseña:
        return render_template('login/registro.html', mensaje="❌ Complete usuario y contraseña.")

    hashed = bcrypt.hashpw(contraseña.encode('utf-8'), bcrypt.gensalt())  # bytes
    # convertimos a str para almacenar en DB (utf-8)
    hashed_str = hashed.decode('utf-8')

    conexion = None
    try:
        conexion = conectar()
        cursor = conexion.cursor()
        cursor.execute("INSERT INTO usuarios (username, password) VALUES (%s, %s)", (usuario, hashed_str))
        conexion.commit()
        cursor.close()
        mensaje = "✅ Usuario registrado correctamente."
    except mysql.connector.Error as err:
        # podrías comprobar err.errno para detectar duplicados (código varía según configuración)
        mensaje = "⚠️ El usuario ya existe o hubo un error en la inserción."
    except Exception as e:
        mensaje = f"⚠️ Error inesperado: {e}"
    finally:
        if conexion:
            conexion.close()

    return render_template('login/registro.html', mensaje=mensaje)


@app.route('/cambiar_password')
def cambiar_password():
    return render_template('login/cambiar_password.html')

@app.route('/actualizar_password', methods=['POST'])
def actualizar_password():
    usuario = request.form.get('username')
    actual = request.form.get('old_password')
    nueva = request.form.get('new_password')
    if not usuario or not actual or not nueva:
        return render_template('login/cambiar_password.html', mensaje="❌ Complete todos los campos.")

    conexion = None
    try:
        conexion = conectar()
        cursor = conexion.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios WHERE username = %s", (usuario,))
        usuario_db = cursor.fetchone()

        if usuario_db and 'password' in usuario_db:
            stored = usuario_db['password']
            stored_bytes = stored.encode('utf-8') if isinstance(stored, str) else stored
            if bcrypt.checkpw(actual.encode('utf-8'), stored_bytes):
                hashed_nueva = bcrypt.hashpw(nueva.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                update_cursor = conexion.cursor()
                update_cursor.execute("UPDATE usuarios SET password = %s WHERE username = %s", (hashed_nueva, usuario))
                conexion.commit()
                update_cursor.close()
                mensaje = "✅ Contraseña actualizada correctamente."
            else:
                mensaje = "❌ Contraseña actual incorrecta."
        else:
            mensaje = "❌ Usuario no encontrado."
        cursor.close()
    except Exception as e:
        mensaje = f"⚠️ Error en la operación: {e}"
    finally:
        if conexion:
            conexion.close()

    return render_template('login/cambiar_password.html', mensaje=mensaje)

@app.route('/logout')
def logout():
    session.clear()  # limpia toda la sesión
    return redirect(url_for('inicio'))  # redirige al inicio

# --- ¡NUEVO! FUNCIÓN Y RUTA PARA EL CHATBOT DE GEMINI ---
def call_gemini_api(user_message):
    """Llama a la API de Gemini para obtener una respuesta del chatbot."""
    # Define la personalidad y las capacidades del bot.
    system_prompt = """
    Eres 'ClimaBot', un asistente experto en planificación de viajes para la aplicación EcoWeather. 
    Tu propósito es ayudar a los usuarios a planificar actividades en PERÚ.
    Tus capacidades son:
    1.  **Sugerir Lugares:** Si un usuario menciona una actividad (ej. "hacer trekking", "ir a la playa", "visitar ruinas"), sugiere 3 lugares excelentes en Perú para esa actividad, describiendo brevemente por qué son buenos.
    2.  **Dar Recomendaciones:** Si el pronóstico del tiempo es adverso (lluvia, frío), proporciona consejos prácticos (ej. "si llueve, lleva un poncho impermeable", "para el frío de la sierra, es bueno abrigarse en capas").
    3.  **Guiar en la App:** Si el usuario pregunta cómo hacer algo, guíalo. Por ejemplo, si dice "quiero comparar", dile que use el enlace 'Comparar' en la barra de navegación.
    Sé siempre amigable, conciso y útil.
    """
    
    # IMPORTANTE: No se requiere una API Key aquí, se asume que el entorno la provee.
    api_key = "AIzaSyBhA4MahNUFlTAGdeClV6q376k-AYbIkUc" 
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": user_message}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status() # Lanza un error si la respuesta no es 2xx
        result = response.json()
        
        # Extraer el texto de la respuesta de Gemini
        candidate = result.get("candidates", [{}])[0]
        content = candidate.get("content", {}).get("parts", [{}])[0]
        return content.get("text", "No pude procesar tu solicitud en este momento.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al llamar a la API de Gemini: {e}")
        return "Hubo un problema de conexión con el asistente. Inténtalo más tarde."
    except (KeyError, IndexError):
        print(f"❌ Error: La respuesta de la API de Gemini no tuvo el formato esperado.")
        return "Recibí una respuesta inesperada del asistente."

@app.route('/api/chatbot', methods=['POST'])
def chatbot_logic():
    data = request.json
    user_message = data.get('message', '').lower()
    
    if not user_message:
        return jsonify({'response': 'Por favor, escribe un mensaje.'})

    # Llama a la nueva función de Gemini
    bot_response = call_gemini_api(user_message)
    
    # Por ahora, no redirigimos automáticamente, Gemini dará las instrucciones.
    return jsonify({'response': bot_response, 'redirect_url': None})




# ---------------- EJECUCIÓN ----------------
if __name__ == '__main__':
    inicializar_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
