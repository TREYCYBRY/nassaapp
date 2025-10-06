import mysql.connector

def conectar():
    try:
        conexion = mysql.connector.connect(
            host="localhost",
            user="root",       # tu usuario de MySQL
            password="",       # tu contraseña de MySQL
            database="login_db"
        )
        return conexion
    except mysql.connector.Error as err:
        print("Error de conexión:", err)
        return None
