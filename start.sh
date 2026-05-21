#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  ParqueaderoUnibague — Script de instalación y arranque
# ═══════════════════════════════════════════════════════════

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "  ██████╗  █████╗ ██████╗  ██████╗ ██╗   ██╗███████╗ ██████╗ "
echo "  ██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██║   ██║██╔════╝██╔═══██╗"
echo "  ██████╔╝███████║██████╔╝██║   ██║██║   ██║█████╗  ██║   ██║"
echo "  ██╔═══╝ ██╔══██║██╔══██╗██║▄▄ ██║██║   ██║██╔══╝  ██║   ██║"
echo "  ██║     ██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████╗╚██████╔╝"
echo "  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝ ╚═════╝ "
echo "                     Sistema de Parqueadero - Unibagué         "
echo -e "${NC}"

# ── Verificar Python ──
echo -e "${YELLOW}[1/5] Verificando Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 no encontrado. Instálalo desde https://python.org${NC}"
    exit 1
fi
PYTHON_VER=$(python3 --version 2>&1)
echo -e "✅ $PYTHON_VER"

# ── Verificar pip ──
echo -e "${YELLOW}[2/5] Verificando pip...${NC}"
python3 -m pip --version > /dev/null 2>&1 || {
    echo -e "${RED}❌ pip no encontrado${NC}"; exit 1;
}
echo -e "✅ pip disponible"

# ── Entorno virtual ──
echo -e "${YELLOW}[3/5] Creando entorno virtual...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "✅ Entorno virtual creado"
else
    echo -e "✅ Entorno virtual ya existe"
fi

# ── Activar venv ──
source venv/bin/activate

# ── Instalar dependencias ──
echo -e "${YELLOW}[4/5] Instalando dependencias (puede tardar varios minutos)...${NC}"
pip install --upgrade pip -q
pip install -r backend/requirements.txt -q
echo -e "✅ Dependencias instaladas"

# ── Verificar modelo ──
echo -e "${YELLOW}[5/5] Verificando modelo YOLOv8...${NC}"
if [ -f "models/best.pt" ]; then
    echo -e "✅ Modelo personalizado encontrado: models/best.pt"
else
    echo -e "${YELLOW}⚠️  No se encontró models/best.pt"
    echo -e "   Se usará YOLOv8n base para detección."
    echo -e "   Copia tu modelo entrenado en: $(pwd)/models/best.pt${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Instalación completada — Iniciando servidor...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  🌐 Interfaz web:  ${YELLOW}http://localhost:8000${NC}"
echo -e "  📚 API docs:      ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "  🛑 Para detener:  Ctrl+C"
echo ""

# ── Iniciar servidor ──
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
