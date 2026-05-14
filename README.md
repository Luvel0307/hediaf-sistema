# 🦶 Sistema Integral de Pie Diabético v5.0

Sistema híbrido que combina **Deep Learning** (MobileNetV2) y **Lógica Difusa Tipo-2** para la evaluación y seguimiento del pie diabético.

## 📋 Basado en

- **IWGDF 2023** - International Working Group on the Diabetic Foot
- **NOM-015-SSA2-2010** - Prevención, tratamiento y control de la diabetes mellitus
- **SSA México** - Protocolos de manejo integral del pie diabético
- **IDSA 2012** - Diabetic Foot Infections

## 🏗️ Arquitectura

```
pie_diabetico_flask/
├── app.py                              # Aplicación Flask principal
├── pdf_generator.py                    # Generación de PDFs (ReportLab)
├── best_model_diabetic_foot_v3.keras   # Modelo DL (MobileNetV2)
├── README.md
├── LICENSE.txt                         # Licencia propietaria de uso restringido
├── INFORME_TECNICO_ANTIBIOTICOS.md     # Informe técnico: protocolo de antibióticos
├── INFORME_TECNICO_ANTIBIOTICOS.pdf    # Versión PDF del informe técnico
├── .gitignore                          # Archivos excluidos del control de versiones
├── requirements.txt
├── instance/
│   └── pie_diabetico.db               # Base de datos SQLite (se crea automáticamente)
├── templates/
│   ├── base.html                       # Template base con navbar y Bootstrap
│   ├── index.html                      # Página de inicio
│   ├── login.html                      # Login
│   ├── register.html                   # Registro de usuarios
│   ├── admin_dashboard.html            # Dashboard administrador
│   ├── medico_dashboard.html           # Dashboard médico
│   ├── medico_evaluar.html             # Formulario de evaluación clínica
│   ├── evaluacion_resultado.html       # Resultado de evaluación
│   ├── medico_pacientes.html           # Gestión de pacientes
│   ├── medico_alertas.html             # Alertas del médico
│   ├── medico_citas.html               # Gestión de citas
│   ├── medico_historial.html           # Historial con gráficas
│   ├── paciente_dashboard.html         # Dashboard del paciente
│   ├── paciente_seguimiento.html       # Seguimiento diario
│   ├── paciente_seguimiento_resultado.html  # Resultado del seguimiento
│   ├── paciente_evolucion.html         # Gráficas de evolución
│   ├── paciente_guia.html             # Guía de cuidados
│   └── paciente_perfil.html           # Edición de perfil del paciente
├── migrate_db.py                       # Script de migración de base de datos
└── static/
    ├── css/style.css                   # Estilos personalizados
    ├── js/main.js                      # JavaScript personalizado
    └── uploads/                        # Imágenes subidas
```

## 👥 Roles del Sistema

### Administrador
- Dashboard global con estadísticas
- Visualización de todos los usuarios, evaluaciones y alertas

### Médico
- **Evaluación clínica** con IA (Deep Learning + Lógica Difusa Tipo-2)
- **Alertas automáticas** cuando un paciente tiene gravedad ≥ 3
- **Gestión de citas** con pacientes
- **Historial** con gráficas de evolución (Chart.js)
- **Sugerencias de antibióticos** según IWGDF 2023 / NOM-015
- **📄 Descarga de Reporte PDF** — Reporte clínico completo con resultado de IA, parámetros, antibióticos e imagen
- **📷 Captura de foto desde cámara** en dispositivos móviles durante evaluación
- Gestión de pacientes asignados

### Paciente
- **Registro con datos clínicos**: Edad, sexo, tipo de diabetes y años desde el diagnóstico
- **Edición de perfil**: Página dedicada para actualizar datos clínicos en cualquier momento
- **Seguimiento diario** simplificado con lenguaje sencillo
- **📷 Captura de foto desde cámara** en dispositivos móviles (iOS/Android)
- **Semáforo visual** de estado (verde, amarillo, naranja, rojo)
- **Gráficas de evolución** temporal
- **Guía de cuidados** del pie diabético
- **📄 Descarga de Resumen PDF** — Resumen simplificado con estado actual, recomendaciones de cuidado y señales de alarma
- **⚠️ SEGURIDAD**: NO recibe recomendaciones de medicamentos, solo consejos de higiene, descarga y alertas para acudir al médico

## 🔬 Componentes Técnicos

### Deep Learning (CNN)
- Modelo: `best_model_diabetic_foot_v3.keras` (MobileNetV2)
- Clasificación binaria: Úlcera Diabética vs. Herida Ordinaria
- Preprocesamiento: 224×224 RGB, normalización /255
- Umbral optimizado: 0.93
- Inversión de probabilidad: True (corrección de etiquetas del dataset)

### Lógica Difusa Tipo-2 Intervalar
- **Variables de entrada**: SignosLocales, EritemaCm, Profundidad, SignosSist, Isquemia, GlucosaMgdl
- **Variable de salida**: Gravedad (1-4)
- Sistemas Lower y Upper con funciones de membresía diferenciadas
- Cálculo de incertidumbre (FOU) y confianza del sistema
- Reglas clínicas basadas en IWGDF 2023

### Sistema de Alertas
- Automáticas al médico cuando Gravedad ≥ 3
- Tanto desde evaluaciones médicas como seguimientos del paciente

### 📄 Generación de PDFs (ReportLab)
- **Reporte Clínico (Médico)**: PDF profesional A4 con encabezado institucional, datos del paciente, resultado de IA, semáforo visual, parámetros clínicos, recomendaciones, tratamiento antibiótico sugerido, resultado Deep Learning e imagen de la lesión
- **Resumen del Paciente**: PDF simplificado con lenguaje sencillo, estado actual (semáforo), recomendaciones de cuidado diario, señales de alarma, historial de seguimientos y frecuencia de consultas recomendada
- Rutas disponibles:
  - `/medico/descargar_reporte/<evaluacion_id>` — Reporte clínico completo
  - `/paciente/descargar_resumen` — Resumen simplificado del paciente
  - `/medico/descargar_reporte_paciente/<paciente_id>` — Última evaluación del paciente

## 🚀 Instalación

### Requisitos
- **Python 3.11 o superior** (compatible con Python 3.12+)
- TensorFlow 2.x (para el modelo DL)
- scikit-fuzzy 0.5.0+ (requerido para compatibilidad con Python 3.12, la versión 0.4.2 **NO** es compatible con Python 3.12 debido a la eliminación del módulo `imp`)

### Pasos

```bash
# 1. Clonar/copiar el proyecto
cd pie_diabetico_flask

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
python app.py
```

### Migración de base de datos existente
Si tienes una versión anterior de la base de datos, ejecuta el script de migración antes de iniciar:
```bash
python migrate_db.py
```
Esto agregará los campos `edad`, `sexo`, `tipo_diabetes` y `anios_diagnostico` a la tabla de pacientes sin perder datos existentes.

### Acceder a la aplicación
Abrir en el navegador: `http://localhost:5000`

### Usuarios de prueba
| Rol | Usuario | Contraseña |
|-----|---------|------------|
| Admin | `admin` | `admin123` |
| Médico | `dr_garcia` | `medico123` |
| Paciente | `paciente1` | `paciente123` |

## 📦 Dependencias

```
Flask==3.0.0
Flask-SQLAlchemy==3.1.1
Flask-Login==0.6.3
scikit-fuzzy>=0.5.0
numpy==1.24.3
pandas==2.0.3
Pillow==10.1.0
Werkzeug==3.0.1
tensorflow>=2.12.0
reportlab>=4.0.0
```

## ⚠️ Notas Importantes

- **Este sistema es de apoyo diagnóstico** y **NO sustituye el criterio clínico**.
- La base de datos SQLite se crea automáticamente al iniciar la aplicación.
- El modelo Keras debe estar presente en la raíz del proyecto.
- Si TensorFlow no está disponible, el sistema funciona solo con Lógica Difusa.
- **Python 3.12+**: Se requiere `scikit-fuzzy>=0.5.0`. La versión 0.4.2 usa el módulo `imp` que fue eliminado en Python 3.12, causando `ModuleNotFoundError: No module named 'imp'`.

## 📚 Referencias

1. IWGDF Guidelines 2023 - Infection Classification & Glucose Criteria
2. NOM-015-SSA2-2010 - Diabetes Mellitus (México)
3. SSA México - Protocolo Pie Diabético
4. Wagner Classification System
5. IDSA 2012 - Diabetic Foot Infections

## 📄 Documentación Técnica y Legal

### Informe Técnico de Antibióticos
El archivo `INFORME_TECNICO_ANTIBIOTICOS.md` (y su versión PDF) contiene la documentación académica completa del protocolo de asignación de antibióticos, incluyendo:
- Marco teórico (IWGDF 2023, NOM-015-SSA2-2010, IDSA 2012)
- Descripción completa del sistema de Lógica Difusa Tipo-2 Intervalar
- Funciones de membresía y reglas de inferencia
- Protocolo detallado de asignación de antibióticos por nivel de gravedad
- Justificación clínica y espectro de cobertura de cada antibiótico
- Ejemplos de casos clínicos
- Referencias bibliográficas completas (formato APA)

### Licencia
El archivo `LICENSE.txt` establece una **licencia propietaria de uso restringido para investigación**, que protege la propiedad intelectual de la autora Guadalupe Vélez Pérez. Consultar el archivo para los términos completos.

## 👩‍💻 Autora

**Guadalupe Vélez Pérez** — © 2026. Todos los derechos reservados.