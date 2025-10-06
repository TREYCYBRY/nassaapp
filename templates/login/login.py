from conexion import conectar

def login():
    conexion = conectar()
    if not conexion:
        return

    cursor = conexion.cursor()

    usuario = input("Usuario: ")
    contraseña = input("Contraseña: ")

    # Consultamos el usuario
    query = "SELECT * FROM usuarios WHERE username = %s AND password = %s"
    cursor.execute(query, (usuario, contraseña))
    resultado = cursor.fetchone()

    if resultado:
        print("\nInicio de sesión exitoso. Bienvenido,", usuario)
    else:
        print("\nUsuario o contraseña incorrectos.")

    cursor.close()
    conexion.close()

# Ejecutar
if __name__ == "__main__":
    login()
