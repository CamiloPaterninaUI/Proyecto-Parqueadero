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
PATRON_PLACA_CARRO = re.compile(r"^[A-Z]{3}[0-9]{3}$")
PATRON_PLACA_MOTO  = re.compile(r"^[A-Z]{3}[0-9]{2}[A-Z]$")

# Correcciones OCR por posición
CONFUSION_EN_LETRAS = {"0":"O","1":"I","2":"Z","4":"A","5":"S","6":"G","8":"B","9":"q"}
CONFUSION_EN_NUMEROS = {"O":"0","o":"0","I":"1","i":"1","l":"1","B":"8","b":"8","S":"5","s":"5","G":"6","g":"6","Z":"2","q":"9","Q":"0"}

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
    return re.sub(r"[\s\.\-_]", "", texto.upper())

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
    candidatos = []
    for t in textos:
        limpio = normalizar_texto_bruto(t)
        if limpio: candidatos.append(limpio)

    total = normalizar_texto_bruto("".join(textos))
    if total: candidatos.append(total)

    solo_alnum = re.sub(r"[^A-Z0-9]", "", total)
    if solo_alnum: candidatos.append(solo_alnum)

    for cand in candidatos:
        for i in range(len(cand) - 5):
            sub = cand[i:i+6]
            placa, tipo = corregir_placa_posicional(sub)
            if placa:
                conf = 0.9 if len(cand) == 6 else 0.7
                return placa, tipo, conf

    return None, None, 0.0

def detectar_tipo_vehiculo(results) -> str:
    if not results or not results[0].boxes:
        return "carro"
    nombres = results[0].names
    for box in results[0].boxes:
        nombre = nombres.get(int(box.cls[0]), "").lower()
        if any(k in nombre for k in ["moto", "motorcycle", "bike", "scooter"]):
            return "moto"
        if any(k in nombre for k in ["car", "carro", "auto", "truck", "bus"]):
            return "carro"
    return "carro"

def preprocesar_para_ocr(img: np.ndarray) -> list[np.ndarray]:
    variantes = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if h < 80:
        scale = 80 // h + 1
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    v1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    variantes.append(v1)

    _, v2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(v2)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, v4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(v4)

    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, v5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(v5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, v6 = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(v6)

    return variantes

def detectar_y_leer_placa(imagen_bytes: bytes) -> dict:
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "No se pudo decodificar la imagen"}

    h, w = img.shape[:2]

    results = yolo_model(img, conf=0.25, verbose=False)
    tipo_vehiculo = detectar_tipo_vehiculo(results)
    confianza_yolo = 0.0
    regiones = []

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            nombre_clase = results[0].names.get(cls_id, "").lower()
            conf = float(box.conf[0])
            if any(p in nombre_clase for p in ["placa", "plate", "license", "matricula"]):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad = 12
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                regiones.append(img[y1:y2, x1:x2])
                confianza_yolo = max(confianza_yolo, conf)

    regiones.append(img)

    placa_detectada = None
    tipo_detectado = tipo_vehiculo
    confianza_ocr = 0.0
    todos_textos_crudos = []

    for region in regiones:
        variantes = preprocesar_para_ocr(region)
        for variante in variantes:
            ocr_results = ocr_reader.readtext(
                variante,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            textos = [t for (_, t, _) in ocr_results]
            confs  = [c for (_, _, c) in ocr_results]
            todos_textos_crudos.extend(textos)

            placa, tipo, conf = buscar_placa_en_textos(textos)
            if placa:
                conf_ocr_promedio = sum(confs) / len(confs) if confs else conf
                if conf_ocr_promedio > confianza_ocr:
                    placa_detectada = placa
                    tipo_detectado  = tipo
                    confianza_ocr   = conf_ocr_promedio

        if placa_detectada:
            break

    return {
        "placa": placa_detectada,
        "tipo_vehiculo": tipo_detectado,
        "confianza_yolo": round(confianza_yolo, 3),
        "confianza_ocr": round(confianza_ocr, 3),
        "texto_crudo_ocr": list(dict.fromkeys(todos_textos_crudos)),
        "detecciones_yolo": len(results[0].boxes) if results and results[0].boxes else 0,
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