"""
ParqueaderoUnibague - Backend Principal
OCR de placas colombianas con YOLOv8 + EasyOCR
API REST con FastAPI
"""

import os
import re
import cv2
import json
import uuid
import time
import base64
import numpy as np
import easyocr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
import uvicorn

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
DB_FILE = BASE_DIR / "registros.json"

UPLOADS_DIR.mkdir(exist_ok=True)

# Tarifas (COP por hora)
TARIFAS = {
    "moto": {"hora": 1500, "fraccion": 800, "nombre": "Motocicleta"},
    "carro": {"hora": 3000, "fraccion": 1500, "nombre": "Automóvil"},
}

# Patrones placa colombiana
# Patrones placa colombiana (Se mantienen igual, están perfectos)
PATRON_PLACA_CARRO = re.compile(r"^[A-Z]{3}[0-9]{3}$")
PATRON_PLACA_MOTO  = re.compile(r"^[A-Z]{3}[0-9]{2}[A-Z]$")

# Correcciones OCR por posición
# ─────────────────────────────────────────────────────────────────────────────
# Cuando esperamos una LETRA pero el OCR devuelve un número o símbolo raro
CONFUSION_EN_LETRAS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
    # Símbolos / ruido
    "q": "Q",
    "ñ": "N", "Ñ": "N",
    "@": "O",
    # Confusiones forma-forma documentadas en placas colombianas amarillas
    # (el fondo amarillo crea halos que distorsionan ciertos caracteres)
    "D": "O",   # D sin el palo de la derecha → O
    "U": "U",   # U es válida → se deja (no confundir)
    # Las que vimos en el ejemplo ZBL→SOG:
    # Z se lee S cuando el OCR ve los dos trazos diagonales como una curva
    # → la corrección va en el sentido contrario: si leemos S en posición letra
    #   Y el resultado final no forma placa, NO hacemos nada (no hay regla S→Z
    #   porque S es letra válida). El fix real está en mejorar el preproceso.
}

# Cuando esperamos un NÚMERO pero el OCR devuelve una letra
CONFUSION_EN_NUMEROS = {
    "O": "0", "o": "0",
    "Q": "0",
    "D": "0",
    "U": "0",
    "I": "1", "i": "1",
    "l": "1", "L": "1",
    "T": "1", "t": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "g": "6",
    "B": "8", "b": "8",
    "q": "9", "P": "9",
    "J": "3",
    "H": "4",   # H de cuatro trazos a veces pierde el trazo horizontal → 4
    "F": "7",   # F sin el trazo inferior → 7
}
# ──────────────────────────────────────────────
# Inicialización modelos
# ──────────────────────────────────────────────
print("🚀 Cargando modelos...")

model_path = MODELS_DIR / "best.pt"
if model_path.exists():
    yolo_model = YOLO(str(model_path))
    print(f"✅ YOLOv8 cargado desde {model_path}")
else:
    yolo_model = YOLO("yolov8n.pt")
    print("⚠️  Usando YOLOv8 base (coloca tu best.pt en /models/)")

ocr_reader = easyocr.Reader(["en"], gpu=False)
print("✅ EasyOCR listo")

# ──────────────────────────────────────────────
# Base de datos JSON simple
# ──────────────────────────────────────────────
def cargar_db() -> dict:
    if DB_FILE.exists():
        with open(DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
            # Aseguramos que la llave reservas exista en DBs viejas
            if "reservas" not in db:
                db["reservas"] = []
            return db
    return {"vehiculos_activos": {}, "historial": [], "reservas": []}

def guardar_db(data: dict):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

# ──────────────────────────────────────────────
# Modelos Pydantic
# ──────────────────────────────────────────────
class VehiculoEntrada(BaseModel):
    placa: str
    tipo: str

class VehiculoSalida(BaseModel):
    placa: str

class ReservaIn(BaseModel):
    placa: str
    tipo: str
    hora_esperada: str

class RegistroResponse(BaseModel):
    success: bool
    mensaje: str
    data: Optional[dict] = None

# ──────────────────────────────────────────────
# Lógica OCR y detección
# ──────────────────────────────────────────────
def corregir_caracter(c: str, es_letra: bool) -> str:
    c = c.upper()
    if es_letra:
        return CONFUSION_EN_LETRAS.get(c, c)
    else:
        return CONFUSION_EN_NUMEROS.get(c, c)

def normalizar_texto_bruto(texto: str) -> str:
    # Elimina espacios, puntos, guiones, el separador • de placas colombianas
    # y cualquier otro carácter no alfanumérico
    return re.sub(r"[^A-Z0-9]", "", texto.upper())

def corregir_placa_posicional(texto: str) -> tuple[str | None, str | None]:
    if len(texto) != 6:
        return None, None

    patron_carro = [True, True, True, False, False, False]
    resultado_carro = "".join(corregir_caracter(c, es_letra) for c, es_letra in zip(texto, patron_carro))
    if PATRON_PLACA_CARRO.match(resultado_carro):
        return resultado_carro, "carro"

    patron_moto = [True, True, True, False, False, True]
    resultado_moto = "".join(corregir_caracter(c, es_letra) for c, es_letra in zip(texto, patron_moto))
    if PATRON_PLACA_MOTO.match(resultado_moto):
        return resultado_moto, "moto"

    return None, None

def buscar_placa_en_textos(textos: list[str]) -> tuple[str | None, str | None, float]:
    """
    Busca un patrón de placa colombiana en los textos del OCR.
    Estrategias en orden de confianza:
    1. Cada texto individual limpio (6 chars exactos → máxima confianza)
    2. Subcadenas de 6 dentro de texto largo
    3. Combinaciones de 2 fragmentos (para cuando el OCR parte la placa en 2)
    """
    candidatos_con_conf: list[tuple[str, float]] = []

    limpios = [normalizar_texto_bruto(t) for t in textos if normalizar_texto_bruto(t)]

    # Estrategia 1 y 2: textos individuales
    for limpio in limpios:
        conf_base = 0.9 if len(limpio) == 6 else (0.75 if len(limpio) > 6 else 0.5)
        for i in range(len(limpio) - 5):
            sub = limpio[i:i+6]
            placa, tipo = corregir_placa_posicional(sub)
            if placa:
                candidatos_con_conf.append((placa, tipo, conf_base if len(limpio) == 6 else 0.7))
                break

    # Estrategia 3: combinaciones de 2 fragmentos (orden AB y BA)
    if len(limpios) >= 2:
        for i in range(len(limpios)):
            for j in range(len(limpios)):
                if i == j:
                    continue
                combinado = limpios[i] + limpios[j]
                solo_alnum = re.sub(r"[^A-Z0-9]", "", combinado)
                for k in range(len(solo_alnum) - 5):
                    sub = solo_alnum[k:k+6]
                    placa, tipo = corregir_placa_posicional(sub)
                    if placa:
                        candidatos_con_conf.append((placa, tipo, 0.6))
                        break

    # También intentar con el texto total concatenado
    total = re.sub(r"[^A-Z0-9]", "", "".join(limpios))
    if total:
        for i in range(len(total) - 5):
            sub = total[i:i+6]
            placa, tipo = corregir_placa_posicional(sub)
            if placa:
                candidatos_con_conf.append((placa, tipo, 0.55))
                break

    if not candidatos_con_conf:
        return None, None, 0.0

    # Retornar el candidato con mayor confianza
    candidatos_con_conf.sort(key=lambda x: x[2], reverse=True)
    placa, tipo, conf = candidatos_con_conf[0]
    return placa, tipo, conf

def detectar_tipo_vehiculo(results, img_shape=None) -> str:
    """
    Determina si el vehículo es moto o carro basado en las detecciones de YOLO.
    1. Primero busca clases explícitas (moto/carro) en los nombres del modelo.
    2. Si el modelo solo detecta placas, usa el aspect ratio del recorte como señal.
       - Placa carro colombiana: ~33×13 cm → aspect ratio ≈ 2.5
       - Placa moto colombiana:  ~20×15 cm → aspect ratio ≈ 1.3
    3. Fallback: 'carro'
    """
    if not results or not results[0].boxes:
        return "carro"

    palabras_moto  = {"moto", "motorcycle", "bike", "scooter", "motocicleta"}
    palabras_carro = {"car", "carro", "auto", "truck", "bus", "van", "suv", "automovil", "automobile"}
    nombres = results[0].names

    # Paso 1: buscar tipo de vehículo explícito en las clases
    tiene_carro = False
    for box in results[0].boxes:
        nombre = nombres.get(int(box.cls[0]), "").lower()
        if any(kw in nombre for kw in palabras_moto):
            return "moto"
        if any(kw in nombre for kw in palabras_carro):
            tiene_carro = True

    if tiene_carro:
        return "carro"

    # Paso 2: si solo hay detecciones de "placa", usar aspect ratio del recorte
    # Placa moto ≈ cuadrada (ratio < 1.8), placa carro ≈ rectangular (ratio >= 1.8)
    for box in results[0].boxes:
        nombre = nombres.get(int(box.cls[0]), "").lower()
        if any(p in nombre for p in ["placa", "plate", "license", "matricula"]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ancho = max(1, x2 - x1)
            alto  = max(1, y2 - y1)
            ratio = ancho / alto
            if ratio < 1.8:
                return "moto"
            else:
                return "carro"

    return "carro"

def escalar_region(img: np.ndarray, alto_objetivo: int = 80) -> np.ndarray:
    """
    Escala la región al alto objetivo manteniendo proporción.
    Usamos INTER_LANCZOS4 para upscaling (preserva bordes de letras mejor que CUBIC).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = alto_objetivo / h
    nuevo_w = max(1, int(w * scale))
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (nuevo_w, alto_objetivo), interpolation=interp)


def recortar_zona_caracteres(img: np.ndarray) -> np.ndarray:
    """
    Para placas colombianas amarillas: detecta el rectángulo amarillo/naranja
    y descarta el borde decorativo y el texto 'COLOMBIA' inferior.
    Si no detecta el color, devuelve la imagen original sin modificar.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rango amarillo colombiano (fondo de placa) – dos rangos porque el amarillo
    # cruza el 0° del tono en HSV
    mask_amarillo = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))

    contornos, _ = cv2.findContours(mask_amarillo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img

    # El contorno más grande debería ser el fondo de la placa
    c = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Validación: debe ocupar al menos 30% del área total
    if w * h < 0.30 * img.shape[0] * img.shape[1]:
        return img

    recorte = img[y:y+h, x:x+w]

    # Descartar el 20% inferior (franja "COLOMBIA" + borde inferior)
    recorte = recorte[:int(h * 0.80), :]
    return recorte


def preprocesar_para_ocr(img: np.ndarray) -> list[np.ndarray]:
    """
    Genera variantes de preprocesamiento pensadas específicamente para placas
    colombianas amarillas con texto negro.

    Orden de variantes: de menor a mayor agresividad.
    EasyOCR suele rendir mejor con texto OSCURO sobre fondo CLARO,
    que es exactamente el caso de las placas colombianas estándar.
    """
    # ── Paso 0: recortar fondo amarillo y escalar ──────────────────────────────
    img = recortar_zona_caracteres(img)
    img = escalar_region(img, alto_objetivo=80)   # altura fija → kernels consistentes

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variantes = []

    # ── V1: Canal de valor HSV con CLAHE ──────────────────────────────────────
    # El canal V en HSV aísla bien el contraste amarillo-negro sin mezclar
    # la información de color que confunde al gris estándar.
    hsv_v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v1 = clahe.apply(hsv_v)
    variantes.append(v1)                          # texto oscuro sobre fondo claro ✓

    # ── V2: Invertida de V1 ───────────────────────────────────────────────────
    # EasyOCR a veces lee mejor texto CLARO sobre OSCURO (depende del modelo).
    variantes.append(cv2.bitwise_not(v1))

    # ── V3: CLAHE en gris + Otsu ──────────────────────────────────────────────
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe2.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
    _, v3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(v3)
    variantes.append(cv2.bitwise_not(v3))         # invertida también

    # ── V4: Bilateral + Otsu (preserva bordes de letras con ruido de compresión) ─
    bilat = cv2.bilateralFilter(gray, 9, 75, 75)
    _, v4 = cv2.threshold(bilat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(v4)

    # ── V5: Umbral adaptativo (maneja sombras cruzadas / iluminación irregular) ─
    v5 = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        11, 6
    )
    variantes.append(v5)

    # ── V6: Morfología sobre V3 (rellena huecos en letras rotas por polvo/daño) ─
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v6 = cv2.morphologyEx(v3, cv2.MORPH_CLOSE, kernel_close)
    variantes.append(v6)

    # ── V7: Sharpening sobre escala de grises pura ────────────────────────────
    # Para placas nítidas, a veces el gris sin umbralizar da mejor resultado.
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    v7 = cv2.filter2D(gray, -1, kernel_sharp)
    variantes.append(v7)

    return variantes


def _ocr_sobre_variantes(
    variantes: list[np.ndarray],
    todos_textos_crudos: list[str],
) -> tuple[str | None, str | None, float]:
    """
    Ejecuta EasyOCR sobre cada variante y devuelve la mejor placa encontrada.
    Retorna (placa, tipo, confianza) o (None, None, 0.0).
    """
    mejor_placa = None
    mejor_tipo  = None
    mejor_conf  = 0.0

    for variante in variantes:
        try:
            ocr_results = ocr_reader.readtext(
                variante,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                min_size=15,
                # paragraph=False asegura que no fusione los caracteres de la placa
                # con texto del entorno (ej: "COLOMBIA", "Freewins")
                paragraph=False,
                # batch_size=4 mejora la velocidad en CPU sin pérdida de precisión
                batch_size=4,
            )
        except Exception:
            continue

        textos = [t for (_, t, _) in ocr_results]
        confs  = [c for (_, _, c) in ocr_results]
        todos_textos_crudos.extend(textos)

        placa, tipo, conf = buscar_placa_en_textos(textos)

        # Usar la confianza del texto que formó la placa, no el promedio global
        if placa:
            # Intentar encontrar la confianza específica del texto más largo
            conf_especifica = conf
            for (_, t, c) in ocr_results:
                limpio = normalizar_texto_bruto(t)
                if placa in limpio or limpio in placa:
                    conf_especifica = max(conf_especifica, c)

            if conf_especifica > mejor_conf:
                mejor_placa = placa
                mejor_tipo  = tipo
                mejor_conf  = conf_especifica

        # Si ya tenemos alta confianza, no seguir procesando variantes
        if mejor_conf >= 0.85:
            break

    return mejor_placa, mejor_tipo, mejor_conf


def detectar_y_leer_placa(imagen_bytes: bytes) -> dict:
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "No se pudo decodificar la imagen"}

    h, w = img.shape[:2]

    # ── 1. Detección YOLO ──────────────────────────────────────────────────────
    results = yolo_model(img, conf=0.25, verbose=False)
    tipo_vehiculo = detectar_tipo_vehiculo(results, img_shape=(h, w))

    regiones: list[tuple[np.ndarray, float]] = []   # (recorte, conf_yolo)
    confianza_yolo = 0.0

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls_id       = int(box.cls[0])
            nombre_clase = results[0].names.get(cls_id, "").lower()
            conf_box     = float(box.conf[0])

            if any(p in nombre_clase for p in ["placa", "plate", "license", "matricula"]):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Padding proporcional al tamaño del recorte
                pad_x = max(8, int((x2 - x1) * 0.05))
                pad_y = max(4, int((y2 - y1) * 0.08))
                x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x);  y2 = min(h, y2 + pad_y)

                recorte = img[y1:y2, x1:x2]
                if recorte.size > 0:
                    regiones.append((recorte, conf_box))
                    confianza_yolo = max(confianza_yolo, conf_box)

    # Ordenar de mayor a menor confianza YOLO para intentar primero los mejores recortes
    regiones.sort(key=lambda x: x[1], reverse=True)

    # ── 2. Fallback: imagen completa si YOLO no encontró placas ───────────────
    if not regiones:
        regiones.append((img, 0.0))

    # ── 3. OCR sobre cada región ───────────────────────────────────────────────
    placa_detectada = None
    tipo_detectado  = tipo_vehiculo
    confianza_ocr   = 0.0
    todos_textos_crudos: list[str] = []

    for recorte, _ in regiones:
        variantes = preprocesar_para_ocr(recorte)
        placa, tipo_ocr, conf = _ocr_sobre_variantes(variantes, todos_textos_crudos)

        if placa and conf > confianza_ocr:
            placa_detectada = placa
            confianza_ocr   = conf
            # El patrón de la placa sabe si es moto (XXX00X) o carro (XXX000);
            # solo sobreescribimos el tipo si YOLO no tenía una clase explícita de vehículo.
            if tipo_vehiculo == "carro" and tipo_ocr:
                tipo_detectado = tipo_ocr

        # Con confianza alta ya no hace falta seguir con otras regiones
        if confianza_ocr >= 0.85:
            break

    return {
        "placa":             placa_detectada,
        "tipo_vehiculo":     tipo_detectado,
        "confianza_yolo":    round(confianza_yolo, 3),
        "confianza_ocr":     round(confianza_ocr, 3),
        "texto_crudo_ocr":   list(set(todos_textos_crudos)),
        "detecciones_yolo":  len(results[0].boxes) if results and results[0].boxes else 0,
    }
# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="ParqueaderoUnibague API",
    description="Sistema de reconocimiento de placas colombianas con YOLOv8 + OCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = BASE_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.delete("/api/reservas/{placa}")
async def eliminar_reserva(placa: str):
    db = cargar_db()
    placa = placa.upper()
    # Filtramos la lista eliminando la que coincida
    db["reservas"] = [r for r in db["reservas"] if not (r["placa"] == placa and r["estado"] == "pendiente")]
    guardar_db(db)
    return {"success": True, "mensaje": f"Reserva {placa} cancelada"}
    
@app.get("/")
async def root():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "ParqueaderoUnibague API v1.0"}

@app.post("/api/ocr/placa")
async def ocr_placa(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    if len(contenido) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen muy grande (máx 10MB)")

    resultado = detectar_y_leer_placa(contenido)

    nombre_archivo = f"{uuid.uuid4().hex}.jpg"
    ruta = UPLOADS_DIR / nombre_archivo
    nparr = np.frombuffer(contenido, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(str(ruta), img)

    resultado["imagen_guardada"] = nombre_archivo
    return JSONResponse(resultado)

@app.post("/api/vehiculo/entrada")
async def registrar_entrada(vehiculo: VehiculoEntrada):
    placa = vehiculo.placa.upper().strip()
    tipo = vehiculo.tipo.lower()

    if tipo not in TARIFAS:
        raise HTTPException(400, f"Tipo inválido. Use: {list(TARIFAS.keys())}")

    db = cargar_db()

    if placa in db["vehiculos_activos"]:
        raise HTTPException(409, f"El vehículo {placa} ya está en el parqueadero")

    ahora = datetime.now()
    registro = {
        "id": uuid.uuid4().hex[:8].upper(),
        "placa": placa,
        "tipo": tipo,
        "nombre_tipo": TARIFAS[tipo]["nombre"],
        "entrada": ahora.isoformat(),
        "salida": None,
        "duracion_minutos": None,
        "valor_cobrado": None,
        "estado": "activo",
    }

    db["vehiculos_activos"][placa] = registro
    guardar_db(db)

    return RegistroResponse(
        success=True,
        mensaje=f"✅ Entrada registrada: {placa} ({TARIFAS[tipo]['nombre']})",
        data={
            **registro,
            "tarifa_hora": TARIFAS[tipo]["hora"],
            "tarifa_fraccion": TARIFAS[tipo]["fraccion"],
        },
    )

@app.post("/api/vehiculo/salida")
async def registrar_salida(vehiculo: VehiculoSalida):
    placa = vehiculo.placa.upper().strip()
    db = cargar_db()

    if placa not in db["vehiculos_activos"]:
        raise HTTPException(404, f"Vehículo {placa} no encontrado en el parqueadero")

    registro = db["vehiculos_activos"][placa]
    ahora = datetime.now()
    entrada = datetime.fromisoformat(registro["entrada"])
    duracion = ahora - entrada
    minutos_totales = int(duracion.total_seconds() / 60)

    tipo = registro["tipo"]
    tarifa = TARIFAS[tipo]

    if minutos_totales <= 30:
        valor = tarifa["fraccion"]
        detalle = "Fracción (≤30 min)"
    else:
        horas_completas = minutos_totales // 60
        minutos_restantes = minutos_totales % 60
        valor = horas_completas * tarifa["hora"]
        if minutos_restantes > 0:
            valor += tarifa["fraccion"]
        detalle = f"{horas_completas}h {minutos_restantes}min"

    registro.update({
        "salida": ahora.isoformat(),
        "duracion_minutos": minutos_totales,
        "valor_cobrado": valor,
        "detalle_cobro": detalle,
        "estado": "pagado",
    })

    del db["vehiculos_activos"][placa]
    db["historial"].append(registro)
    guardar_db(db)

    horas_str = f"{minutos_totales // 60}h {minutos_totales % 60}min"

    return RegistroResponse(
        success=True,
        mensaje=f"💰 Cobro: ${valor:,.0f} COP — {TARIFAS[tipo]['nombre']} {placa}",
        data={
            **registro,
            "duracion_str": horas_str,
            "detalle_cobro": detalle,
            "tarifa_aplicada": tarifa,
        },
    )

@app.get("/api/vehiculos/activos")
async def listar_activos():
    db = cargar_db()
    ahora = datetime.now()
    activos = []

    for placa, reg in db["vehiculos_activos"].items():
        entrada = datetime.fromisoformat(reg["entrada"])
        minutos = int((ahora - entrada).total_seconds() / 60)
        tipo = reg["tipo"]
        tarifa = TARIFAS[tipo]

        if minutos <= 30:
            cobro_actual = tarifa["fraccion"]
        else:
            h = minutos // 60
            m = minutos % 60
            cobro_actual = h * tarifa["hora"] + (tarifa["fraccion"] if m > 0 else 0)

        activos.append({
            **reg,
            "minutos_transcurridos": minutos,
            "duracion_str": f"{minutos // 60}h {minutos % 60}min",
            "cobro_actual": cobro_actual,
        })

    activos.sort(key=lambda x: x["entrada"])
    return {"total": len(activos), "vehiculos": activos}

@app.get("/api/historial")
async def obtener_historial(limite: int = 50):
    db = cargar_db()
    historial = db["historial"][-limite:]
    historial.reverse()
    return {
        "total": len(db["historial"]),
        "registros": historial,
    }

@app.get("/api/estadisticas")
async def estadisticas():
    db = cargar_db()
    hoy = datetime.now().date().isoformat()

    ingresos_hoy = 0
    vehiculos_hoy = 0
    motos_hoy = 0
    carros_hoy = 0

    for reg in db["historial"]:
        if reg.get("salida", "")[:10] == hoy:
            vehiculos_hoy += 1
            ingresos_hoy += reg.get("valor_cobrado", 0) or 0
            if reg["tipo"] == "moto":
                motos_hoy += 1
            else:
                carros_hoy += 1

    return {
        "fecha": hoy,
        "activos_ahora": len(db["vehiculos_activos"]),
        "vehiculos_hoy": vehiculos_hoy,
        "ingresos_hoy": ingresos_hoy,
        "motos_hoy": motos_hoy,
        "carros_hoy": carros_hoy,
        "tarifas_vigentes": TARIFAS,
    }

@app.get("/api/tarifas")
async def obtener_tarifas():
    return TARIFAS

@app.post("/api/reservas")
async def crear_reserva(reserva: ReservaIn):
    placa = reserva.placa.upper().strip()
    tipo = reserva.tipo.lower()
    
    if tipo not in TARIFAS:
        raise HTTPException(400, "Tipo de vehículo inválido")
        
    db = cargar_db()
    
    if placa in db["vehiculos_activos"]:
        raise HTTPException(409, "El vehículo ya está dentro del parqueadero")

    # Verificar que no exista ya una reserva pendiente para esta placa
    ya_reservado = any(r["placa"] == placa and r["estado"] == "pendiente" for r in db["reservas"])
    if ya_reservado:
        raise HTTPException(409, f"Ya existe una reserva pendiente para {placa}")
        
    nueva_reserva = {
        "id": uuid.uuid4().hex[:8].upper(),
        "placa": placa,
        "tipo": tipo,
        "nombre_tipo": TARIFAS[tipo]["nombre"],
        "hora_esperada": reserva.hora_esperada,
        "estado": "pendiente",
        "creado_en": datetime.now().isoformat()
    }
    
    db["reservas"].append(nueva_reserva)
    guardar_db(db)
    
    return {"success": True, "mensaje": f"✅ Reserva creada para {placa}"}

@app.get("/api/reservas")
async def listar_reservas():
    db = cargar_db()
    pendientes = [r for r in db["reservas"] if r["estado"] == "pendiente"]
    return {"total": len(pendientes), "reservas": pendientes}

@app.post("/api/reservas/{placa}/llegada")
async def confirmar_llegada_reserva(placa: str):
    db = cargar_db()
    placa = placa.upper()
    
    reserva = next((r for r in db["reservas"] if r["placa"] == placa and r["estado"] == "pendiente"), None)
    if not reserva:
        raise HTTPException(404, "No hay reserva pendiente para esta placa")
        
    reserva["estado"] = "completada"
    guardar_db(db)
    
    return await registrar_entrada(VehiculoEntrada(placa=reserva["placa"], tipo=reserva["tipo"]))

# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)