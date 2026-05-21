Proyecto Realizado por Camilo Paternina y Alejandro Silva
Estudiantes de la Universidad de Ibagué - Ingeniería de Sistemas

# 🚗 Sistema Inteligente de Parqueadero

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![YOLO](https://img.shields.io/badge/YOLO-AI%20Detection-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Estado-En%20Desarrollo-orange?style=for-the-badge)

---

# 📌 Descripción

Sistema inteligente de gestión de parqueadero desarrollado con Python, visión artificial e inteligencia artificial para detectar y administrar espacios de estacionamiento en tiempo real.

El proyecto utiliza un modelo entrenado (`best.pt`) para realizar detección automática y control de espacios disponibles dentro del parqueadero.

---

# ✨ Características

✅ Detección automática de vehículos  
✅ Monitoreo de espacios en tiempo real  
✅ Interfaz visual amigable  
✅ Uso de Inteligencia Artificial con modelo YOLO  
✅ Sistema rápido y eficiente  
✅ Diseño adaptable y moderno  

---

# 🛠️ Tecnologías Utilizadas

- 🐍 Python
- 👁️ OpenCV
- 🤖 YOLO / Ultralytics
- 🌐 HTML
- 🎨 CSS
- ⚡ JavaScript

---

# 📂 Estructura del Proyecto

```bash
├───backend
│   ├───uploads
│   └───__pycache__
├───frontend
├───models
└───venv
    ├───Include
    ├───Lib
    │   └───site-packages
    │       ├───pip
    │       │   ├───_internal
    │       │   │   ├───cli
    │       │   │   │   └───__pycache__
    │       │   │   ├───commands
    │       │   │   │   └───__pycache__
    │       │   │   ├───distributions
    │       │   │   │   └───__pycache__
    │       │   │   ├───index
    │       │   │   │   └───__pycache__
    │       │   │   ├───locations
    │       │   │   │   └───__pycache__
    │       │   │   ├───metadata
    │       │   │   │   ├───importlib
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───models
    │       │   │   │   └───__pycache__
    │       │   │   ├───network
    │       │   │   │   └───__pycache__
    │       │   │   ├───operations
    │       │   │   │   ├───build
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───install
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───req
    │       │   │   │   └───__pycache__
    │       │   │   ├───resolution
    │       │   │   │   ├───legacy
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───resolvelib
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───utils
    │       │   │   │   └───__pycache__
    │       │   │   ├───vcs
    │       │   │   │   └───__pycache__
    │       │   │   └───__pycache__
    │       │   ├───_vendor
    │       │   │   ├───cachecontrol
    │       │   │   │   ├───caches
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───certifi
    │       │   │   │   └───__pycache__
    │       │   │   ├───chardet
    │       │   │   │   ├───cli
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───metadata
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───colorama
    │       │   │   │   └───__pycache__
    │       │   │   ├───distlib
    │       │   │   │   └───__pycache__
    │       │   │   ├───distro
    │       │   │   │   └───__pycache__
    │       │   │   ├───idna
    │       │   │   │   └───__pycache__
    │       │   │   ├───msgpack
    │       │   │   │   └───__pycache__
    │       │   │   ├───packaging
    │       │   │   │   └───__pycache__
    │       │   │   ├───pep517
    │       │   │   │   ├───in_process
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───pkg_resources
    │       │   │   │   └───__pycache__
    │       │   │   ├───platformdirs
    │       │   │   │   └───__pycache__
    │       │   │   ├───pygments
    │       │   │   │   ├───filters
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───formatters
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───lexers
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───styles
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───pyparsing
    │       │   │   │   ├───diagram
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───requests
    │       │   │   │   └───__pycache__
    │       │   │   ├───resolvelib
    │       │   │   │   ├───compat
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───rich
    │       │   │   │   └───__pycache__
    │       │   │   ├───tenacity
    │       │   │   │   └───__pycache__
    │       │   │   ├───tomli
    │       │   │   │   └───__pycache__
    │       │   │   ├───urllib3
    │       │   │   │   ├───contrib
    │       │   │   │   │   ├───_securetransport
    │       │   │   │   │   │   └───__pycache__
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───packages
    │       │   │   │   │   ├───backports
    │       │   │   │   │   │   └───__pycache__
    │       │   │   │   │   └───__pycache__
    │       │   │   │   ├───util
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───webencodings
    │       │   │   │   └───__pycache__
    │       │   │   └───__pycache__
    │       │   └───__pycache__
    │       ├───pip-22.3.1.dist-info
    │       ├───pkg_resources
    │       │   ├───extern
    │       │   │   └───__pycache__
    │       │   ├───_vendor
    │       │   │   ├───importlib_resources
    │       │   │   │   └───__pycache__
    │       │   │   ├───jaraco
    │       │   │   │   ├───text
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───more_itertools
    │       │   │   │   └───__pycache__
    │       │   │   ├───packaging
    │       │   │   │   └───__pycache__
    │       │   │   ├───pyparsing
    │       │   │   │   ├───diagram
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   └───__pycache__
    │       │   └───__pycache__
    │       ├───setuptools
    │       │   ├───command
    │       │   │   └───__pycache__
    │       │   ├───config
    │       │   │   ├───_validate_pyproject
    │       │   │   │   └───__pycache__
    │       │   │   └───__pycache__
    │       │   ├───extern
    │       │   │   └───__pycache__
    │       │   ├───_distutils
    │       │   │   ├───command
    │       │   │   │   └───__pycache__
    │       │   │   └───__pycache__
    │       │   ├───_vendor
    │       │   │   ├───importlib_metadata
    │       │   │   │   └───__pycache__
    │       │   │   ├───importlib_resources
    │       │   │   │   └───__pycache__
    │       │   │   ├───jaraco
    │       │   │   │   ├───text
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───more_itertools
    │       │   │   │   └───__pycache__
    │       │   │   ├───packaging
    │       │   │   │   └───__pycache__
    │       │   │   ├───pyparsing
    │       │   │   │   ├───diagram
    │       │   │   │   │   └───__pycache__
    │       │   │   │   └───__pycache__
    │       │   │   ├───tomli
    │       │   │   │   └───__pycache__
    │       │   │   └───__pycache__
    │       │   └───__pycache__
    │       ├───setuptools-65.5.0.dist-info
    │       └───_distutils_hack
    │           └───__pycache__
    └───Scripts

🚀 Instalación y Ejecución
1️⃣ Clonar el repositorio
git clone https://github.com/CamiloPaterninaUI/Proyecto-Parqueadero.git
2️⃣ Entrar al proyecto
cd Proyecto-Parqueadero
3️⃣ Crear entorno virtual (Opcional pero recomendado)
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
4️⃣ Instalar dependencias
pip install ultralytics opencv-python
5️⃣ Ejecutar el proyecto
python main.py
🧠 Modelo de Inteligencia Artificial

El archivo:

best.pt

corresponde al modelo entrenado utilizado para la detección de vehículos y espacios dentro del parqueadero.

👨‍💻 Creadores
Camilo Paternina
Alejandro Silva

