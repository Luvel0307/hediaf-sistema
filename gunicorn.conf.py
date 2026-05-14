# =============================================================
# HeDiaF — Configuración de Gunicorn para Render.com
# =============================================================
import os

# Bind
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Workers — Free tier tiene 512 MB RAM
# 2 workers con 2 threads = 4 conexiones concurrentes
workers = 2
threads = 2

# Timeout largo para carga del modelo DL
timeout = 120
graceful_timeout = 30

# Preload app para compartir modelo entre workers
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Limitar tamaño de request
limit_request_line = 8190
limit_request_fields = 100
