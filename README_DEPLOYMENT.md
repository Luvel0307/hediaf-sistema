# 🚀 Guía de Deployment — HeDiaF en Render.com

## Sistema Integral de Pie Diabético v5.1

Esta guía explica paso a paso cómo desplegar la aplicación en **Render.com** (tier gratuito).

---

## 📋 Checklist Pre-Deployment

- [ ] Cuenta creada en [Render.com](https://render.com)
- [ ] Cuenta creada en [GitHub](https://github.com)
- [ ] Git instalado en tu computadora
- [ ] Todos los archivos del proyecto listos

---

## 📁 Estructura del Proyecto

```
hediaf_deploy/
├── app.py                          # Aplicación principal Flask
├── pdf_generator.py                # Generador de reportes PDF
├── init_db.py                      # Script de inicialización de BD
├── migrate_db.py                   # Migración de BD existente
├── best_model_diabetic_foot_v3.keras  # Modelo Deep Learning (11 MB)
├── requirements.txt                # Dependencias Python
├── build.sh                        # Script de build para Render
├── Procfile                        # Comando de inicio (gunicorn)
├── render.yaml                     # Configuración automática de Render
├── .gitignore                      # Archivos ignorados por Git
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/.gitkeep
├── templates/                      # Plantillas HTML
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── admin_dashboard.html
│   ├── medico_*.html
│   └── paciente_*.html
├── README.md                       # Documentación del proyecto
├── README_DEPLOYMENT.md            # Esta guía
└── LICENSE.txt
```

---

## 🔧 Paso 1: Crear Repositorio en GitHub

### Opción A: Desde la terminal (recomendado)

```bash
# 1. Ir a la carpeta del proyecto
cd hediaf_deploy

# 2. Inicializar Git
git init
git add .
git commit -m "🚀 HeDiaF v5.1 - Listo para Render"

# 3. Crear repositorio en GitHub (desde github.com/new)
#    Nombre sugerido: hediaf-pie-diabetico
#    Dejar PRIVADO si contiene datos sensibles

# 4. Conectar y subir
git remote add origin https://github.com/TU_USUARIO/hediaf-pie-diabetico.git
git branch -M main
git push -u origin main
```

### Opción B: Subir archivos manualmente

1. Ve a [github.com/new](https://github.com/new)
2. Crea un repositorio nuevo llamado `hediaf-pie-diabetico`
3. Sube todos los archivos arrastrándolos

---

## 🌐 Paso 2: Crear Cuenta en Render

1. Ve a [render.com](https://render.com)
2. Clic en **"Get Started for Free"**
3. Regístrate con tu cuenta de **GitHub** (recomendado para conexión directa)
4. Verifica tu email

---

## 🗄️ Paso 3: Crear Base de Datos PostgreSQL

1. En el dashboard de Render, clic en **"New +"** → **"PostgreSQL"**
2. Configura:
   - **Name:** `hediaf-db`
   - **Database:** `hediaf_pie_diabetico`
   - **User:** `hediaf_user`
   - **Region:** Oregon (US West)
   - **Plan:** **Free**
3. Clic en **"Create Database"**
4. ⏳ Espera a que se cree (1-2 minutos)
5. **COPIA** el valor de **"Internal Database URL"** — lo necesitarás

> ⚠️ **Nota:** La BD gratuita de Render se borra después de 90 días de inactividad.

---

## 🖥️ Paso 4: Crear Servicio Web

### Opción A: Deployment Automático con render.yaml (Recomendado)

1. En Render dashboard → **"New +"** → **"Blueprint"**
2. Conecta tu repositorio de GitHub
3. Render detectará `render.yaml` y creará automáticamente:
   - Servicio web
   - Base de datos PostgreSQL
   - Variables de entorno
4. Clic en **"Apply"**

### Opción B: Deployment Manual

1. En Render dashboard → **"New +"** → **"Web Service"**
2. Conecta tu repositorio de GitHub
3. Configura:
   - **Name:** `hediaf-pie-diabetico`
   - **Region:** Oregon
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** `./build.sh`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`
   - **Plan:** **Free**

4. En **Environment Variables**, agrega:

   | Key | Value |
   |-----|-------|
   | `DATABASE_URL` | *(pega la Internal Database URL del paso 3)* |
   | `SECRET_KEY` | *(genera una clave aleatoria larga)* |
   | `RENDER` | `true` |
   | `PYTHON_VERSION` | `3.11.9` |
   | `TF_CPP_MIN_LOG_LEVEL` | `2` |
   | `TF_ENABLE_ONEDNN_OPTS` | `0` |

5. Clic en **"Create Web Service"**

---

## ⏳ Paso 5: Esperar el Build

El primer build tarda **10-15 minutos** porque:
- Instala TensorFlow (~400 MB)
- Compila dependencias numéricas
- Inicializa la base de datos

Puedes ver el progreso en la pestaña **"Logs"** de Render.

### ✅ Señales de éxito en los logs:
```
✅ Sistema Difuso Tipo-2 inicializado
✅ Base de datos inicializada
🦶 SISTEMA INTEGRAL DE PIE DIABÉTICO — HeDiaF
   Entorno: PRODUCCIÓN (Render)
```

---

## ✅ Paso 6: Verificar que Funciona

1. Render te dará una URL como: `https://hediaf-pie-diabetico.onrender.com`
2. Abre la URL en tu navegador
3. Deberías ver la página de inicio del sistema
4. Prueba iniciar sesión con:
   - **Admin:** `admin` / `admin123`
   - **Médico:** `dr_garcia` / `medico123`
   - **Paciente:** `paciente1` / `paciente123`

---

## 🔍 Checklist Post-Deployment

- [ ] Página de inicio carga correctamente
- [ ] Login funciona para los 3 roles
- [ ] Dashboard de médico muestra correctamente
- [ ] Evaluación con lógica difusa funciona
- [ ] Subida de imagen funciona
- [ ] Análisis Deep Learning funciona (puede tardar la primera vez)
- [ ] Generación de PDF funciona
- [ ] Dashboard de paciente funciona
- [ ] Seguimiento de paciente funciona
- [ ] Registro de nuevo usuario funciona
- [ ] Alertas se generan correctamente

---

## 🐛 Solución de Problemas

### Error: "Application failed to respond"
- Revisa los **Logs** en Render
- El modelo DL se carga lazy; la primera evaluación con imagen puede tardar ~30s
- Si el build falla, verifica `requirements.txt`

### Error: "Database connection refused"
- Verifica que `DATABASE_URL` esté configurado correctamente
- Asegúrate de usar la **Internal Database URL** (no la External)

### Error de memoria (512 MB)
- El tier gratuito tiene 512 MB RAM
- El modelo Keras usa ~200 MB
- Si hay problemas, revisa si tienes muchos workers en gunicorn

### El sitio tarda en cargar (cold start)
- En el plan gratuito, Render apaga el servicio después de 15 min de inactividad
- La primera visita después de inactividad tarda ~30-60 segundos (cold start)
- Esto es normal en el tier gratuito

### Las imágenes subidas se pierden
- En Render free, el filesystem no es persistente
- Las imágenes subidas se borran en cada deploy
- Para producción real, usa un servicio de almacenamiento (S3, Cloudinary)

---

## 📝 Notas Importantes

1. **Tier Gratuito:** El servicio se duerme después de 15 min sin tráfico. La primera visita tarda más.
2. **Base de datos:** Se mantiene 90 días en inactividad. Después se borra.
3. **Modelo DL:** Se incluye en el repositorio (~11 MB). Se carga en la primera evaluación con imagen.
4. **Archivos subidos:** NO son persistentes en Render Free. Se pierden en cada redeploy.
5. **SECRET_KEY:** Usar siempre la generada por Render en producción.

---

## 🔄 Actualizaciones Futuras

Para actualizar la aplicación:

```bash
# Hacer cambios en tu código local
git add .
git commit -m "📝 Descripción del cambio"
git push origin main
```

Render detectará el push automáticamente y hará un nuevo deploy.

---

## 📞 Soporte

- **Documentación Render:** [docs.render.com](https://docs.render.com)
- **Estado de Render:** [status.render.com](https://status.render.com)

---

*Sistema de apoyo diagnóstico basado en IWGDF 2023, NOM-015-SSA2-2010, SSA México.*
*No sustituye el criterio clínico.*
