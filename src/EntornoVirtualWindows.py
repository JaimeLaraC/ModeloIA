import os
import platform
import subprocess

def create_venv_and_install_packages():
    # Crear el entorno virtual .venv
    os.system("python -m venv .venv")  # Usa 'python' en lugar de 'python3' en Windows

    # Verificar el sistema operativo para activar el entorno
    if platform.system() == "Windows":
        activate_script = ".venv\\Scripts\\activate.bat"
        # Ejecutar los comandos en el entorno virtual
        command = f"{activate_script} && pip install tensorflow numpy pandas tensorflow_datasets"
        subprocess.call(command, shell=True)
    else:
        activate_script = ".venv/bin/activate"
        # Ejecutar los comandos en el entorno virtual con la opción --break-system-packages
        command = f"source {activate_script} && pip install tensorflow numpy pandas tensorflow_datasets --break-system-packages"
        subprocess.call(command, shell=True, executable="/bin/bash")

    print("Entorno virtual '.venv' creado y paquetes instalados: tensorflow, numpy, pandas.")

create_venv_and_install_packages()
