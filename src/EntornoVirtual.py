import os
import platform
import subprocess
import requests

def create_venv_and_install_packages():
    # Crear el entorno virtual .venv
    os.system("python3 -m venv .venv")
    print("Prueba")

    # Verificar el sistema operativo para activar el entorno
    if platform.system() == "Windows":
        activate_script = ".venv\\Scripts\\activate"
    else:
        activate_script = ".venv/bin/activate"

    # Ejecutar los comandos en el entorno virtual con la opción --break-system-packages
    command = f"source {activate_script} && pip install tensorflow numpy pandas tensorflow_datasets --break-system-packages"
    subprocess.call(command, shell=True, executable="/bin/bash")

    print("Entorno virtual '.venv' creado y paquetes instalados: tensorflow, numpy, pandas.")

create_venv_and_install_packages()
