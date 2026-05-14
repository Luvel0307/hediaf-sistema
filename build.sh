#!/usr/bin/env bash
# =============================================================
# HeDiaF — Script de build para Render.com
# =============================================================
set -o errexit   # Salir si algún comando falla

echo "🔧 Actualizando pip..."
pip install --upgrade pip

echo "📦 Instalando dependencias..."
pip install -r requirements.txt

echo "✅ Build completado"
