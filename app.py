# -*- coding: utf-8 -*-
"""
=============================================================
SISTEMA INTEGRAL DE PIE DIABÉTICO - Flask Application
=============================================================
Versión: 5.0 — Flask Web Application
Combina:
  • Deep Learning (CNN best_model_diabetic_foot_v3.keras / MobileNetV2)
  • Lógica Difusa Tipo-2 Intervalar (Lower/Upper FOU)
  • Sistema de Roles (Admin, Médico, Paciente)
  • Alertas automáticas cuando Gravedad >= 3

Basado en: IWGDF 2023, IDSA, NOM-015-SSA2-2010, SSA México
=============================================================
"""

import os
import json
import numpy as np
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, session, send_from_directory, send_file)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ── Configuración de la aplicación ──────────────────────────
app = Flask(__name__)

# --- Detección de entorno ---
IS_PRODUCTION = os.environ.get('RENDER', '') == 'true' or os.environ.get('DATABASE_URL', '')

# --- SECRET_KEY ---
app.config['SECRET_KEY'] = os.environ.get(
    'SECRET_KEY', 'pie-diabetico-sistema-hibrido-2024-secret-dev'
)

# --- Base de datos: PostgreSQL (Render) o SQLite (local) ---
database_url = os.environ.get('DATABASE_URL', '')
if database_url:
    # Render usa "postgres://" pero SQLAlchemy requiere "postgresql://"
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pie_diabetico.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 300,
    'pool_pre_ping': True,
}

# --- Uploads ---
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max (optimizado para free tier)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor inicia sesión para acceder.'

# ── Modelos de Base de Datos ────────────────────────────────

class Usuario(UserMixin, db.Model):
    __tablename__ = 'usuarios'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    nombre_completo = db.Column(db.String(200), nullable=False)
    rol = db.Column(db.String(20), nullable=False, default='paciente')  # admin, medico, paciente
    cedula_profesional = db.Column(db.String(50))
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow)
    activo = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Paciente(db.Model):
    __tablename__ = 'pacientes'
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=True)
    nombre = db.Column(db.String(200), nullable=False)
    edad = db.Column(db.Integer)
    sexo = db.Column(db.String(20))
    tipo_diabetes = db.Column(db.String(50))
    anios_diagnostico = db.Column(db.Integer)
    medico_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=True)
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow)
    notas = db.Column(db.Text)

    usuario = db.relationship('Usuario', foreign_keys=[usuario_id], backref='paciente_perfil')
    medico = db.relationship('Usuario', foreign_keys=[medico_id], backref='pacientes_asignados')
    evaluaciones = db.relationship('Evaluacion', backref='paciente', lazy='dynamic',
                                   order_by='Evaluacion.fecha.desc()')
    seguimientos = db.relationship('Seguimiento', backref='paciente', lazy='dynamic',
                                    order_by='Seguimiento.fecha.desc()')


class Evaluacion(db.Model):
    __tablename__ = 'evaluaciones'
    id = db.Column(db.Integer, primary_key=True)
    paciente_id = db.Column(db.Integer, db.ForeignKey('pacientes.id'), nullable=False)
    medico_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)

    # Parámetros clínicos
    signos_locales = db.Column(db.Integer, default=0)
    eritema_cm = db.Column(db.Float, default=0.0)
    profundidad = db.Column(db.Integer, default=0)
    signos_sist = db.Column(db.Integer, default=0)
    isquemia = db.Column(db.Integer, default=0)
    glucosa_mgdl = db.Column(db.Float, default=100.0)

    # Resultados difusos
    gravedad_lower = db.Column(db.Float)
    gravedad_upper = db.Column(db.Float)
    gravedad_crisp = db.Column(db.Float)
    incertidumbre = db.Column(db.Float)
    confianza = db.Column(db.String(20))
    grado = db.Column(db.Integer)
    etiqueta = db.Column(db.String(100))
    recomendacion = db.Column(db.Text)

    # Resultado DL
    imagen_path = db.Column(db.String(500))
    dl_es_diabetica = db.Column(db.Boolean)
    dl_probabilidad = db.Column(db.Float)
    dl_categoria = db.Column(db.String(100))

    notas_clinicas = db.Column(db.Text)

    medico = db.relationship('Usuario', backref='evaluaciones_realizadas')


class Seguimiento(db.Model):
    __tablename__ = 'seguimientos'
    id = db.Column(db.Integer, primary_key=True)
    paciente_id = db.Column(db.Integer, db.ForeignKey('pacientes.id'), nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)

    # Datos del paciente (simplificados)
    signos_locales = db.Column(db.Integer, default=0)
    eritema_cm = db.Column(db.Float, default=0.0)
    profundidad = db.Column(db.Integer, default=0)
    signos_sist = db.Column(db.Integer, default=0)
    isquemia = db.Column(db.Integer, default=0)
    glucosa_mgdl = db.Column(db.Float, default=100.0)
    temperatura = db.Column(db.Float)
    dolor_nivel = db.Column(db.Integer)

    # Resultado
    grado = db.Column(db.Integer)
    etiqueta = db.Column(db.String(100))
    gravedad_crisp = db.Column(db.Float)

    notas = db.Column(db.Text)
    imagen_path = db.Column(db.String(500))


class Cita(db.Model):
    __tablename__ = 'citas'
    id = db.Column(db.Integer, primary_key=True)
    paciente_id = db.Column(db.Integer, db.ForeignKey('pacientes.id'), nullable=False)
    medico_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=False)
    fecha_cita = db.Column(db.DateTime, nullable=False)
    motivo = db.Column(db.String(500))
    estado = db.Column(db.String(20), default='pendiente')  # pendiente, completada, cancelada
    notas = db.Column(db.Text)
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)

    paciente = db.relationship('Paciente', backref='citas')
    medico = db.relationship('Usuario', backref='citas_medico')


class Alerta(db.Model):
    __tablename__ = 'alertas'
    id = db.Column(db.Integer, primary_key=True)
    paciente_id = db.Column(db.Integer, db.ForeignKey('pacientes.id'), nullable=False)
    medico_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=True)
    tipo = db.Column(db.String(50), nullable=False)  # gravedad_alta, seguimiento_critico
    mensaje = db.Column(db.Text, nullable=False)
    grado = db.Column(db.Integer)
    leida = db.Column(db.Boolean, default=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)

    paciente = db.relationship('Paciente', backref='alertas')
    medico = db.relationship('Usuario', backref='alertas_recibidas')


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Usuario, int(user_id))


# ── Utilidades ──────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def role_required(*roles):
    """Decorador para restringir acceso por rol"""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if current_user.rol not in roles:
                flash('No tienes permiso para acceder a esta sección.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def crear_alerta(paciente_id, medico_id, tipo, mensaje, grado):
    """Crea una alerta automática"""
    alerta = Alerta(
        paciente_id=paciente_id,
        medico_id=medico_id,
        tipo=tipo,
        mensaje=mensaje,
        grado=grado
    )
    db.session.add(alerta)
    db.session.commit()


# ── Sistema Difuso Tipo-2 ──────────────────────────────────

import skfuzzy as fuzz
from skfuzzy import control as ctrl

class SistemaDifusoTipo2:
    """
    Sistema Experto Difuso Tipo-2 Intervalar para evaluación de
    gravedad de infección en pie diabético.
    Basado en: IWGDF 2023, IDSA, NOM-015-SSA2-2010
    """

    def __init__(self):
        self.FUZZY_LOWER = {}
        self.FUZZY_UPPER = {}
        self.SYS_LOWER = None
        self.SYS_UPPER = None
        self._crear_variables()
        self._definir_funciones_membresia()
        self._construir_reglas()
        print("✅ Sistema Difuso Tipo-2 inicializado")

    def _crear_variables(self):
        universos = {
            'SignosLocales': np.arange(0, 5, 0.1),
            'EritemaCm': np.arange(0, 5.1, 0.1),
            'Profundidad': np.arange(0, 3, 0.1),
            'SignosSist': np.arange(0, 2, 0.1),
            'Isquemia': np.arange(0, 3, 0.1),
            'GlucosaMgdl': np.arange(70, 501, 1),
            'Gravedad': np.arange(1, 5.1, 0.1)
        }
        for key, universe in universos.items():
            tipo = ctrl.Consequent if key == 'Gravedad' else ctrl.Antecedent
            kwargs = {'defuzzify_method': 'centroid'} if key == 'Gravedad' else {}
            self.FUZZY_LOWER[key] = tipo(universe, key, **kwargs)
            self.FUZZY_UPPER[key] = tipo(universe, key, **kwargs)

    def _definir_funciones_membresia(self):
        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            sl = FV['SignosLocales']
            if tag == 'lower':
                sl['ninguno'] = fuzz.trimf(sl.universe, [0, 0, 0.8])
                sl['pocos'] = fuzz.trimf(sl.universe, [0, 1, 1.8])
                sl['varios'] = fuzz.trimf(sl.universe, [1.8, 3, 3.8])
                sl['muchos'] = fuzz.trimf(sl.universe, [3.2, 4, 4])
            else:
                sl['ninguno'] = fuzz.trimf(sl.universe, [0, 0, 1.2])
                sl['pocos'] = fuzz.trimf(sl.universe, [0, 1, 2.2])
                sl['varios'] = fuzz.trimf(sl.universe, [1.5, 3, 4])
                sl['muchos'] = fuzz.trimf(sl.universe, [2.8, 4, 4])

        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            er = FV['EritemaCm']
            if tag == 'lower':
                er['minimo'] = fuzz.trapmf(er.universe, [0, 0, 0.2, 0.4])
                er['pequeno'] = fuzz.trimf(er.universe, [0.3, 1.0, 1.8])
                er['grande'] = fuzz.trapmf(er.universe, [1.9, 2.2, 5, 5])
            else:
                er['minimo'] = fuzz.trapmf(er.universe, [0, 0, 0.4, 0.6])
                er['pequeno'] = fuzz.trimf(er.universe, [0.2, 1.0, 2.2])
                er['grande'] = fuzz.trapmf(er.universe, [1.6, 1.9, 5, 5])

        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            pr = FV['Profundidad']
            d = 0.4 if tag == 'lower' else 0.6
            pr['sin_herida'] = fuzz.trimf(pr.universe, [0, 0, d])
            pr['superficial'] = fuzz.trimf(pr.universe, [1 - d, 1, 1 + d])
            pr['profunda'] = fuzz.trimf(pr.universe, [2 - d, 2, 2])

        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            ss = FV['SignosSist']
            d = 0.4 if tag == 'lower' else 0.6
            ss['no'] = fuzz.trimf(ss.universe, [0, 0, d])
            ss['si'] = fuzz.trimf(ss.universe, [1 - d, 1, 1])

        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            isq = FV['Isquemia']
            d = 0.4 if tag == 'lower' else 0.6
            isq['ninguna'] = fuzz.trimf(isq.universe, [0, 0, d])
            isq['leve'] = fuzz.trimf(isq.universe, [1 - d, 1, 1 + d])
            isq['alta'] = fuzz.trimf(isq.universe, [2 - d, 2, 2])

        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            gl = FV['GlucosaMgdl']
            if tag == 'lower':
                gl['normal'] = fuzz.trapmf(gl.universe, [70, 70, 110, 130])
                gl['elevada'] = fuzz.trimf(gl.universe, [120, 180, 250])
                gl['muy_elevada'] = fuzz.trapmf(gl.universe, [240, 270, 500, 500])
            else:
                gl['normal'] = fuzz.trapmf(gl.universe, [70, 70, 120, 145])
                gl['elevada'] = fuzz.trimf(gl.universe, [110, 180, 265])
                gl['muy_elevada'] = fuzz.trapmf(gl.universe, [225, 255, 500, 500])

        for tag, FV in [('lower', self.FUZZY_LOWER), ('upper', self.FUZZY_UPPER)]:
            gv = FV['Gravedad']
            if tag == 'lower':
                gv['baja'] = fuzz.trimf(gv.universe, [1, 1, 1.8])
                gv['leve'] = fuzz.trimf(gv.universe, [1.6, 2, 2.4])
                gv['moderada'] = fuzz.trimf(gv.universe, [2.3, 3, 3.4])
                gv['grave'] = fuzz.trimf(gv.universe, [3.3, 4, 4])
            else:
                gv['baja'] = fuzz.trimf(gv.universe, [1, 1, 2.0])
                gv['leve'] = fuzz.trimf(gv.universe, [1.5, 2, 2.6])
                gv['moderada'] = fuzz.trimf(gv.universe, [2.2, 3, 3.6])
                gv['grave'] = fuzz.trimf(gv.universe, [3.2, 4, 4])

    def _construir_reglas(self):
        def hacer_reglas(FV):
            sl = FV['SignosLocales']
            er = FV['EritemaCm']
            pr = FV['Profundidad']
            ss = FV['SignosSist']
            isq = FV['Isquemia']
            gl = FV['GlucosaMgdl']
            gv = FV['Gravedad']
            return [
                ctrl.Rule(sl['ninguno'] & ss['no'] & gl['normal'], gv['baja']),
                ctrl.Rule(sl['pocos'] & er['minimo'] & pr['sin_herida'] & ss['no'], gv['baja']),
                ctrl.Rule(sl['varios'] & er['pequeno'] & pr['superficial'] & ss['no'] & isq['ninguna'], gv['leve']),
                ctrl.Rule(sl['pocos'] & er['pequeno'] & pr['superficial'] & ss['no'], gv['leve']),
                ctrl.Rule(er['grande'] & ss['no'], gv['moderada']),
                ctrl.Rule(pr['profunda'] & ss['no'] & sl['varios'], gv['moderada']),
                ctrl.Rule(sl['muchos'] & er['pequeno'] & ss['no'], gv['moderada']),
                ctrl.Rule(gl['muy_elevada'] & sl['varios'] & ss['no'], gv['moderada']),
                ctrl.Rule(ss['si'] | isq['alta'], gv['grave']),
                ctrl.Rule(isq['alta'] & pr['profunda'], gv['grave']),
                ctrl.Rule(ss['si'] & er['grande'], gv['grave']),
                ctrl.Rule(gl['muy_elevada'] & ss['si'], gv['grave']),
                ctrl.Rule(gl['muy_elevada'] & isq['alta'], gv['grave']),
            ]

        self.SYS_LOWER = ctrl.ControlSystem(hacer_reglas(self.FUZZY_LOWER))
        self.SYS_UPPER = ctrl.ControlSystem(hacer_reglas(self.FUZZY_UPPER))

    def evaluar(self, signos_locales, eritema_cm, profundidad, signos_sist, isquemia, glucosa_mgdl):
        inputs = {
            'SignosLocales': int(np.clip(signos_locales, 0, 4)),
            'EritemaCm': float(np.clip(eritema_cm, 0, 5)),
            'Profundidad': int(np.clip(profundidad, 0, 2)),
            'SignosSist': int(np.clip(signos_sist, 0, 1)),
            'Isquemia': int(np.clip(isquemia, 0, 2)),
            'GlucosaMgdl': float(np.clip(glucosa_mgdl, 70, 500))
        }

        resultados = {}
        for nombre, sistema in [('lower', self.SYS_LOWER), ('upper', self.SYS_UPPER)]:
            try:
                sim = ctrl.ControlSystemSimulation(sistema)
                for k, v in inputs.items():
                    sim.input[k] = v
                sim.compute()
                resultados[f'gravedad_{nombre}'] = float(np.clip(sim.output['Gravedad'], 1.0, 4.0))
            except Exception:
                resultados[f'gravedad_{nombre}'] = self._valor_defecto(inputs)

        rL = resultados['gravedad_lower']
        rU = resultados['gravedad_upper']
        rC = (rL + rU) / 2.0
        incertidumbre = abs(rU - rL)
        confianza = "ALTA" if incertidumbre < 0.3 else "MEDIA" if incertidumbre < 0.6 else "BAJA"

        grado, etiqueta, recomendacion, color = self._clasificar(rC, inputs)

        return {
            'gravedad_lower': rL, 'gravedad_upper': rU, 'gravedad_crisp': rC,
            'incertidumbre': incertidumbre, 'confianza': confianza,
            'grado': grado, 'etiqueta': etiqueta, 'recomendacion': recomendacion,
            'color': color, 'inputs': inputs
        }

    def _valor_defecto(self, inputs):
        if inputs['SignosSist'] == 1 or inputs['Isquemia'] == 2:
            return 4.0
        elif inputs['GlucosaMgdl'] > 250 and inputs['SignosLocales'] >= 3:
            return 3.5
        elif inputs['SignosLocales'] >= 3:
            return 3.0
        elif inputs['SignosLocales'] >= 2:
            return 2.0
        return 1.5

    def _clasificar(self, rC, inputs):
        gl = inputs['GlucosaMgdl']
        ss = inputs['SignosSist']
        isq = inputs['Isquemia']
        pr = inputs['Profundidad']
        sl = inputs['SignosLocales']
        er = inputs['EritemaCm']

        hiperglucemia_grave = gl > 250

        criterio_grave = (
            ss == 1 or isq == 2 or
            (hiperglucemia_grave and ss == 1) or
            (hiperglucemia_grave and isq == 2)
        )
        criterio_moderada = (
            er >= 2.0 or pr == 2 or
            (sl >= 3 and er >= 2.0) or
            (hiperglucemia_grave and sl >= 2 and ss == 0)
        )
        criterio_leve = (sl >= 2 and er < 2.0 and pr < 2 and ss == 0)

        if criterio_grave or rC >= 3.5:
            return 4, "Infección grave", (
                "TRASLADO INMEDIATO A URGENCIAS\n"
                "- Riesgo de sepsis y amputación\n"
                "- Requiere hospitalización urgente\n"
                "- Antibióticos IV de amplio espectro\n"
                "- Evaluación quirúrgica inmediata\n"
                "- Control glucémico intensivo"
            ), '#d32f2f'
        elif criterio_moderada:
            return 3, "Infección moderada", (
                "VALORACIÓN MÉDICA HOY MISMO\n"
                "- Acudir al médico en las próximas horas\n"
                "- Probable necesidad de antibióticos IV\n"
                "- Posible hospitalización\n"
                "- Valorar desbridamiento\n"
                "- Optimizar control glucémico"
            ), '#f57c00'
        elif criterio_leve:
            return 2, "Infección leve", (
                "CONSULTA MÉDICA EN 24-48 HORAS\n"
                "- Antibióticos orales según indicación médica\n"
                "- Limpieza diaria con solución salina\n"
                "- Apósito estéril\n"
                "- Vigilar signos de empeoramiento"
            ), '#fbc02d'
        else:
            return 1, "No infectado", (
                "CONTROL Y VIGILANCIA RUTINARIA\n"
                "- Limpieza diaria con solución salina\n"
                "- Apósito estéril no oclusivo\n"
                "- Revisión médica de rutina\n"
                "- Educación en cuidado del pie"
            ), '#388e3c'


# ── Sistema Deep Learning ──────────────────────────────────

modelo_dl = None
sistema_difuso = None

def cargar_modelo_dl():
    """Carga el modelo Keras - compatible con Keras 3.10+ y NumPy 2.x"""
    global modelo_dl
    if modelo_dl is not None:
        return modelo_dl

    modelo_path = os.path.join(app.root_path, 'best_model_diabetic_foot_v3.keras')

    if not os.path.exists(modelo_path):
        print(f"⚠️ Archivo de modelo no encontrado: {modelo_path}")
        return None

    errores = []

    # Intento 1: Carga directa con keras (Keras 3.10+ soporta quantization_config)
    try:
        import keras
        modelo_dl = keras.models.load_model(modelo_path, compile=False)
        print(f"✅ Modelo DL cargado correctamente con keras {keras.__version__}")
        return modelo_dl
    except Exception as e:
        errores.append(f"keras: {e}")

    # Intento 2: Carga con tf.keras
    try:
        import tensorflow as tf
        modelo_dl = tf.keras.models.load_model(modelo_path, compile=False)
        print(f"✅ Modelo DL cargado correctamente con tf.keras (TF {tf.__version__})")
        return modelo_dl
    except Exception as e:
        errores.append(f"tf.keras: {e}")

    # Intento 3 (fallback): Reconstruir arquitectura y cargar pesos
    try:
        import keras
        import zipfile
        import tempfile

        base = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base.trainable = False

        modelo = keras.Sequential([
            keras.layers.InputLayer(shape=(224, 224, 3)),
            base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Extraer pesos del archivo .keras (es un ZIP)
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(modelo_path, 'r') as z:
            z.extract('model.weights.h5', tmp_dir)
        weights_path = os.path.join(tmp_dir, 'model.weights.h5')
        modelo.load_weights(weights_path)

        modelo_dl = modelo
        print("✅ Modelo DL reconstruido y cargado correctamente (fallback)")
        return modelo_dl
    except Exception as e:
        errores.append(f"fallback: {e}")

    print(f"⚠️ Error cargando modelo DL: {' | '.join(errores)}")
    return None

def clasificar_imagen(imagen_path, umbral=0.93, invertir_prob=True):
    """Clasifica una imagen usando el modelo DL (MobileNetV2 224x224)"""
    modelo = cargar_modelo_dl()
    if modelo is None:
        return {'error': 'Modelo no disponible', 'es_diabetica': False,
                'probabilidad_diabetica': 0.0, 'categoria': 'Sin modelo'}

    try:
        from PIL import Image
        img = Image.open(imagen_path).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prob_raw = float(modelo.predict(img_array, verbose=0)[0][0])
        prob_diabetica = (1 - prob_raw) if invertir_prob else prob_raw
        es_diabetica = prob_diabetica > umbral
        confianza = prob_diabetica if es_diabetica else (1 - prob_diabetica)
        categoria = "Úlcera Diabética" if es_diabetica else "Herida Ordinaria"

        return {
            'es_diabetica': es_diabetica,
            'confianza': confianza,
            'probabilidad_diabetica': prob_diabetica,
            'probabilidad_raw': prob_raw,
            'categoria': categoria,
            'umbral_usado': umbral
        }
    except Exception as e:
        return {'error': str(e), 'es_diabetica': False,
                'probabilidad_diabetica': 0.0, 'categoria': 'Error'}


def inicializar_sistema_difuso():
    """Inicializa el sistema difuso"""
    global sistema_difuso
    if sistema_difuso is None:
        sistema_difuso = SistemaDifusoTipo2()
    return sistema_difuso


# ── RUTAS: Autenticación ────────────────────────────────────

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.rol == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif current_user.rol == 'medico':
            return redirect(url_for('medico_dashboard'))
        else:
            return redirect(url_for('paciente_dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = Usuario.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f'Bienvenido, {user.nombre_completo}', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        flash('Usuario o contraseña incorrectos.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        nombre = request.form.get('nombre_completo', '').strip()
        rol = request.form.get('rol', 'paciente')
        cedula = request.form.get('cedula_profesional', '').strip()

        if Usuario.query.filter_by(username=username).first():
            flash('Este nombre de usuario ya existe.', 'danger')
            return render_template('register.html')
        if Usuario.query.filter_by(email=email).first():
            flash('Este correo ya está registrado.', 'danger')
            return render_template('register.html')

        user = Usuario(username=username, email=email, nombre_completo=nombre,
                       rol=rol, cedula_profesional=cedula if rol == 'medico' else None)
        user.set_password(password)
        db.session.add(user)

        # Si es paciente, crear perfil de paciente automáticamente
        if rol == 'paciente':
            db.session.flush()  # para obtener user.id
            edad = request.form.get('edad', type=int)
            tipo_diabetes = request.form.get('tipo_diabetes', '').strip()
            sexo = request.form.get('sexo', '').strip()
            anios_diagnostico = request.form.get('anios_diagnostico', type=int)
            pac = Paciente(
                usuario_id=user.id, nombre=nombre,
                edad=edad, sexo=sexo,
                tipo_diabetes=tipo_diabetes,
                anios_diagnostico=anios_diagnostico
            )
            db.session.add(pac)

        db.session.commit()
        flash('Registro exitoso. Inicia sesión.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Sesión cerrada correctamente.', 'info')
    return redirect(url_for('login'))


# ── RUTAS: Admin ────────────────────────────────────────────

@app.route('/admin/dashboard')
@role_required('admin')
def admin_dashboard():
    total_usuarios = Usuario.query.count()
    total_pacientes = Paciente.query.count()
    total_evaluaciones = Evaluacion.query.count()
    total_seguimientos = Seguimiento.query.count()
    alertas_activas = Alerta.query.filter_by(leida=False).count()
    ultimas_evaluaciones = Evaluacion.query.order_by(Evaluacion.fecha.desc()).limit(10).all()
    medicos = Usuario.query.filter_by(rol='medico').all()
    return render_template('admin_dashboard.html',
                           total_usuarios=total_usuarios,
                           total_pacientes=total_pacientes,
                           total_evaluaciones=total_evaluaciones,
                           total_seguimientos=total_seguimientos,
                           alertas_activas=alertas_activas,
                           ultimas_evaluaciones=ultimas_evaluaciones,
                           medicos=medicos)


# ── RUTAS: Médico ──────────────────────────────────────────

@app.route('/medico/dashboard')
@role_required('medico')
def medico_dashboard():
    mis_pacientes = Paciente.query.filter_by(medico_id=current_user.id).all()
    alertas = Alerta.query.filter_by(medico_id=current_user.id, leida=False)\
                          .order_by(Alerta.fecha.desc()).all()
    citas_pendientes = Cita.query.filter_by(medico_id=current_user.id, estado='pendiente')\
                                  .order_by(Cita.fecha_cita.asc()).all()
    evaluaciones_recientes = Evaluacion.query.filter_by(medico_id=current_user.id)\
                                              .order_by(Evaluacion.fecha.desc()).limit(5).all()
    return render_template('medico_dashboard.html',
                           mis_pacientes=mis_pacientes,
                           alertas=alertas,
                           citas_pendientes=citas_pendientes,
                           evaluaciones_recientes=evaluaciones_recientes)


@app.route('/medico/pacientes')
@role_required('medico')
def medico_pacientes():
    mis_pacientes = Paciente.query.filter_by(medico_id=current_user.id).all()
    todos_pacientes = Paciente.query.filter_by(medico_id=None).all()
    return render_template('medico_pacientes.html',
                           mis_pacientes=mis_pacientes,
                           pacientes_sin_asignar=todos_pacientes)


@app.route('/medico/asignar_paciente/<int:paciente_id>', methods=['POST'])
@role_required('medico')
def asignar_paciente(paciente_id):
    pac = db.session.get(Paciente, paciente_id)
    if pac:
        pac.medico_id = current_user.id
        db.session.commit()
        flash(f'Paciente {pac.nombre} asignado correctamente.', 'success')
    return redirect(url_for('medico_pacientes'))


@app.route('/medico/evaluar', methods=['GET', 'POST'])
@app.route('/medico/evaluar/<int:paciente_id>', methods=['GET', 'POST'])
@role_required('medico')
def medico_evaluar(paciente_id=None):
    mis_pacientes = Paciente.query.filter_by(medico_id=current_user.id).all()
    paciente_seleccionado = db.session.get(Paciente, paciente_id) if paciente_id else None

    if request.method == 'POST':
        pac_id = request.form.get('paciente_id', type=int)
        paciente = db.session.get(Paciente, pac_id)
        if not paciente:
            flash('Paciente no encontrado.', 'danger')
            return redirect(url_for('medico_evaluar'))

        # Obtener parámetros clínicos
        signos_locales = request.form.get('signos_locales', 0, type=int)
        eritema_cm = request.form.get('eritema_cm', 0.0, type=float)
        profundidad = request.form.get('profundidad', 0, type=int)
        signos_sist = request.form.get('signos_sist', 0, type=int)
        isquemia_val = request.form.get('isquemia', 0, type=int)
        glucosa = request.form.get('glucosa_mgdl', 100.0, type=float)
        notas = request.form.get('notas_clinicas', '')

        # Evaluar con sistema difuso
        sdf = inicializar_sistema_difuso()
        resultado = sdf.evaluar(signos_locales, eritema_cm, profundidad,
                                signos_sist, isquemia_val, glucosa)

        # Procesar imagen si se subió
        imagen_filename = None
        dl_resultado = None
        if 'imagen' in request.files:
            file = request.files['imagen']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"{pac_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                imagen_filename = filename
                dl_resultado = clasificar_imagen(filepath)

        # Guardar evaluación
        ev = Evaluacion(
            paciente_id=pac_id, medico_id=current_user.id,
            signos_locales=signos_locales, eritema_cm=eritema_cm,
            profundidad=profundidad, signos_sist=signos_sist,
            isquemia=isquemia_val, glucosa_mgdl=glucosa,
            gravedad_lower=resultado['gravedad_lower'],
            gravedad_upper=resultado['gravedad_upper'],
            gravedad_crisp=resultado['gravedad_crisp'],
            incertidumbre=resultado['incertidumbre'],
            confianza=resultado['confianza'],
            grado=resultado['grado'], etiqueta=resultado['etiqueta'],
            recomendacion=resultado['recomendacion'],
            imagen_path=imagen_filename,
            dl_es_diabetica=dl_resultado['es_diabetica'] if dl_resultado and 'error' not in dl_resultado else None,
            dl_probabilidad=dl_resultado['probabilidad_diabetica'] if dl_resultado and 'error' not in dl_resultado else None,
            dl_categoria=dl_resultado['categoria'] if dl_resultado else None,
            notas_clinicas=notas
        )
        db.session.add(ev)

        # ALERTA AUTOMÁTICA si gravedad >= 3
        if resultado['grado'] >= 3:
            crear_alerta(
                paciente_id=pac_id,
                medico_id=current_user.id,
                tipo='gravedad_alta',
                mensaje=f"⚠️ Paciente {paciente.nombre}: {resultado['etiqueta']} "
                        f"(Grado {resultado['grado']}/4, Crisp: {resultado['gravedad_crisp']:.2f}). "
                        f"Glucosa: {glucosa:.0f} mg/dL. Requiere atención prioritaria.",
                grado=resultado['grado']
            )

        db.session.commit()

        # Construir antibióticos sugeridos (solo para médicos)
        antibioticos = {
            1: "No requiere antibióticos",
            2: "Cefalexina 500 mg c/6h VO × 7-14 días\nAlternativa: Amoxicilina-clavulánico 875/125 mg c/12h VO",
            3: "Amoxicilina-clavulánico 1 g c/8h IV o VO\nAlternativa: Clindamicina 600 mg c/8h + Ciprofloxacino 400 mg c/12h IV",
            4: "Piperacilina-tazobactam 4.5 g c/6h IV\nAlternativa: Meropenem 1 g c/8h IV + Vancomicina 15-20 mg/kg c/12h IV"
        }

        return render_template('evaluacion_resultado.html',
                               paciente=paciente, resultado=resultado,
                               evaluacion=ev, dl_resultado=dl_resultado,
                               antibioticos=antibioticos.get(resultado['grado'], ''))

    return render_template('medico_evaluar.html',
                           mis_pacientes=mis_pacientes,
                           paciente_seleccionado=paciente_seleccionado)


@app.route('/medico/alertas')
@role_required('medico')
def medico_alertas():
    alertas = Alerta.query.filter_by(medico_id=current_user.id)\
                          .order_by(Alerta.fecha.desc()).all()
    return render_template('medico_alertas.html', alertas=alertas)


@app.route('/medico/alerta/leer/<int:alerta_id>', methods=['POST'])
@role_required('medico')
def marcar_alerta_leida(alerta_id):
    alerta = db.session.get(Alerta, alerta_id)
    if alerta and alerta.medico_id == current_user.id:
        alerta.leida = True
        db.session.commit()
    return redirect(url_for('medico_alertas'))


@app.route('/medico/citas', methods=['GET', 'POST'])
@role_required('medico')
def medico_citas():
    if request.method == 'POST':
        pac_id = request.form.get('paciente_id', type=int)
        fecha_str = request.form.get('fecha_cita', '')
        motivo = request.form.get('motivo', '')
        try:
            fecha_cita = datetime.strptime(fecha_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            flash('Formato de fecha inválido.', 'danger')
            return redirect(url_for('medico_citas'))

        cita = Cita(paciente_id=pac_id, medico_id=current_user.id,
                     fecha_cita=fecha_cita, motivo=motivo)
        db.session.add(cita)
        db.session.commit()
        flash('Cita creada correctamente.', 'success')

    mis_pacientes = Paciente.query.filter_by(medico_id=current_user.id).all()
    citas = Cita.query.filter_by(medico_id=current_user.id)\
                      .order_by(Cita.fecha_cita.desc()).all()
    return render_template('medico_citas.html', citas=citas, mis_pacientes=mis_pacientes)


@app.route('/medico/cita/completar/<int:cita_id>', methods=['POST'])
@role_required('medico')
def completar_cita(cita_id):
    cita = db.session.get(Cita, cita_id)
    if cita and cita.medico_id == current_user.id:
        cita.estado = 'completada'
        cita.notas = request.form.get('notas', '')
        db.session.commit()
        flash('Cita marcada como completada.', 'success')
    return redirect(url_for('medico_citas'))


@app.route('/medico/historial/<int:paciente_id>')
@role_required('medico')
def medico_historial(paciente_id):
    paciente = db.session.get(Paciente, paciente_id)
    if not paciente:
        flash('Paciente no encontrado.', 'danger')
        return redirect(url_for('medico_dashboard'))
    evaluaciones = paciente.evaluaciones.all()
    seguimientos = paciente.seguimientos.all()

    # Datos para gráficas
    eval_data = []
    for e in evaluaciones:
        eval_data.append({
            'fecha': e.fecha.strftime('%Y-%m-%d %H:%M'),
            'grado': e.grado,
            'gravedad_crisp': round(e.gravedad_crisp, 2) if e.gravedad_crisp else 0,
            'glucosa': e.glucosa_mgdl or 0
        })

    seg_data = []
    for s in seguimientos:
        seg_data.append({
            'fecha': s.fecha.strftime('%Y-%m-%d %H:%M'),
            'grado': s.grado,
            'gravedad_crisp': round(s.gravedad_crisp, 2) if s.gravedad_crisp else 0,
            'glucosa': s.glucosa_mgdl or 0
        })

    chart_data = json.dumps(eval_data + seg_data)

    return render_template('medico_historial.html',
                           paciente=paciente,
                           evaluaciones=evaluaciones,
                           seguimientos=seguimientos,
                           chart_data=chart_data)


# ── RUTAS: Paciente ────────────────────────────────────────

@app.route('/paciente/dashboard')
@role_required('paciente')
def paciente_dashboard():
    pac = Paciente.query.filter_by(usuario_id=current_user.id).first()
    if not pac:
        pac = Paciente(usuario_id=current_user.id, nombre=current_user.nombre_completo)
        db.session.add(pac)
        db.session.commit()

    ultimo_seguimiento = pac.seguimientos.first()
    total_seguimientos = pac.seguimientos.count()
    citas = Cita.query.filter_by(paciente_id=pac.id, estado='pendiente')\
                      .order_by(Cita.fecha_cita.asc()).all()
    alertas = Alerta.query.filter_by(paciente_id=pac.id, leida=False)\
                          .order_by(Alerta.fecha.desc()).limit(5).all()

    return render_template('paciente_dashboard.html',
                           paciente=pac,
                           ultimo_seguimiento=ultimo_seguimiento,
                           total_seguimientos=total_seguimientos,
                           citas=citas,
                           alertas=alertas)


@app.route('/paciente/seguimiento', methods=['GET', 'POST'])
@role_required('paciente')
def paciente_seguimiento():
    pac = Paciente.query.filter_by(usuario_id=current_user.id).first()
    if not pac:
        flash('No se encontró tu perfil de paciente.', 'danger')
        return redirect(url_for('paciente_dashboard'))

    if request.method == 'POST':
        signos_locales = request.form.get('signos_locales', 0, type=int)
        eritema_cm = request.form.get('eritema_cm', 0.0, type=float)
        profundidad = request.form.get('profundidad', 0, type=int)
        signos_sist = request.form.get('signos_sist', 0, type=int)
        isquemia_val = request.form.get('isquemia', 0, type=int)
        glucosa = request.form.get('glucosa_mgdl', 100.0, type=float)
        temperatura = request.form.get('temperatura', type=float)
        dolor = request.form.get('dolor_nivel', 0, type=int)
        notas = request.form.get('notas', '')

        sdf = inicializar_sistema_difuso()
        resultado = sdf.evaluar(signos_locales, eritema_cm, profundidad,
                                signos_sist, isquemia_val, glucosa)

        # Procesar imagen opcional
        imagen_filename = None
        if 'imagen' in request.files:
            file = request.files['imagen']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"seg_{pac.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                imagen_filename = filename

        seg = Seguimiento(
            paciente_id=pac.id,
            signos_locales=signos_locales, eritema_cm=eritema_cm,
            profundidad=profundidad, signos_sist=signos_sist,
            isquemia=isquemia_val, glucosa_mgdl=glucosa,
            temperatura=temperatura, dolor_nivel=dolor,
            grado=resultado['grado'], etiqueta=resultado['etiqueta'],
            gravedad_crisp=resultado['gravedad_crisp'],
            notas=notas, imagen_path=imagen_filename
        )
        db.session.add(seg)

        # ALERTA si gravedad >= 3
        if resultado['grado'] >= 3 and pac.medico_id:
            crear_alerta(
                paciente_id=pac.id,
                medico_id=pac.medico_id,
                tipo='seguimiento_critico',
                mensaje=f"🚨 Seguimiento de {pac.nombre}: {resultado['etiqueta']} "
                        f"(Grado {resultado['grado']}/4). "
                        f"Glucosa: {glucosa:.0f} mg/dL. El paciente necesita atención.",
                grado=resultado['grado']
            )

        db.session.commit()

        # Para pacientes: recomendaciones SIN medicamentos
        consejos_paciente = {
            1: {
                'estado': '🟢 Tu pie está bien',
                'que_hacer': [
                    'Sigue lavando tus pies diariamente con agua tibia y jabón suave',
                    'Seca bien entre los dedos',
                    'Aplica crema hidratante (no entre los dedos)',
                    'Revisa tus pies cada día buscando heridas o cambios',
                    'Usa zapatos cómodos y calcetines de algodón'
                ],
                'alerta': False
            },
            2: {
                'estado': '🟡 Tu pie necesita atención',
                'que_hacer': [
                    'Agenda una cita con tu médico en las próximas 24-48 horas',
                    'Limpia la herida con solución salina (NO uses alcohol ni agua oxigenada)',
                    'Cubre con un apósito estéril limpio',
                    'No camines descalzo',
                    'Vigila si el enrojecimiento crece o aparece fiebre'
                ],
                'alerta': False
            },
            3: {
                'estado': '🟠 ¡Atención! Tu pie necesita valoración médica urgente',
                'que_hacer': [
                    '⚠️ ACUDE AL MÉDICO HOY MISMO',
                    'No toques ni intentes curar la herida por tu cuenta',
                    'Mantén el pie elevado y en reposo',
                    'Si tienes fiebre o escalofríos, ve a urgencias',
                    'Lleva este registro a tu médico'
                ],
                'alerta': True
            },
            4: {
                'estado': '🔴 ¡URGENCIA! Necesitas atención médica inmediata',
                'que_hacer': [
                    '🚨 VE A URGENCIAS AHORA',
                    'No esperes, pide que te lleven al hospital',
                    'Tu pie tiene señales de infección grave',
                    'Necesitas tratamiento hospitalario',
                    'Lleva este registro al médico de urgencias'
                ],
                'alerta': True
            }
        }

        consejo = consejos_paciente.get(resultado['grado'], consejos_paciente[1])

        return render_template('paciente_seguimiento_resultado.html',
                               resultado=resultado, consejo=consejo,
                               seguimiento=seg, paciente=pac)

    return render_template('paciente_seguimiento.html', paciente=pac)


@app.route('/paciente/evolucion')
@role_required('paciente')
def paciente_evolucion():
    pac = Paciente.query.filter_by(usuario_id=current_user.id).first()
    if not pac:
        flash('No se encontró tu perfil.', 'danger')
        return redirect(url_for('paciente_dashboard'))

    seguimientos = pac.seguimientos.order_by(Seguimiento.fecha.asc()).all()

    chart_data = []
    for s in seguimientos:
        chart_data.append({
            'fecha': s.fecha.strftime('%Y-%m-%d %H:%M'),
            'grado': s.grado,
            'gravedad_crisp': round(s.gravedad_crisp, 2) if s.gravedad_crisp else 0,
            'glucosa': s.glucosa_mgdl or 0
        })

    return render_template('paciente_evolucion.html',
                           paciente=pac, seguimientos=seguimientos,
                           chart_data=json.dumps(chart_data))


@app.route('/paciente/perfil', methods=['GET', 'POST'])
@role_required('paciente')
def paciente_perfil():
    pac = Paciente.query.filter_by(usuario_id=current_user.id).first()
    if not pac:
        flash('No se encontró tu perfil.', 'danger')
        return redirect(url_for('paciente_dashboard'))

    if request.method == 'POST':
        pac.edad = request.form.get('edad', type=int)
        pac.sexo = request.form.get('sexo', '').strip()
        pac.tipo_diabetes = request.form.get('tipo_diabetes', '').strip()
        pac.anios_diagnostico = request.form.get('anios_diagnostico', type=int)
        pac.nombre = request.form.get('nombre', pac.nombre).strip()
        db.session.commit()
        flash('Perfil actualizado correctamente.', 'success')
        return redirect(url_for('paciente_perfil'))

    return render_template('paciente_perfil.html', paciente=pac)


@app.route('/paciente/guia')
@role_required('paciente')
def paciente_guia():
    return render_template('paciente_guia.html')


# ── RUTAS: Descarga de PDFs ─────────────────────────────────

from pdf_generator import generar_pdf_reporte_medico, generar_pdf_resumen_paciente


@app.route('/medico/descargar_reporte/<int:evaluacion_id>')
@role_required('medico')
def descargar_reporte_medico(evaluacion_id):
    """Genera y descarga el PDF del reporte clínico de una evaluación."""
    evaluacion = db.session.get(Evaluacion, evaluacion_id)
    if not evaluacion:
        flash('Evaluación no encontrada.', 'danger')
        return redirect(url_for('medico_dashboard'))
    if evaluacion.medico_id != current_user.id:
        flash('No tienes permiso para descargar este reporte.', 'danger')
        return redirect(url_for('medico_dashboard'))

    paciente = db.session.get(Paciente, evaluacion.paciente_id)
    medico = db.session.get(Usuario, evaluacion.medico_id)

    pdf_buffer = generar_pdf_reporte_medico(
        evaluacion=evaluacion,
        paciente=paciente,
        medico=medico,
        upload_folder=app.config['UPLOAD_FOLDER']
    )

    filename = f"Reporte_Clinico_{paciente.nombre.replace(' ', '_')}_{evaluacion.fecha.strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


@app.route('/paciente/descargar_resumen')
@role_required('paciente')
def descargar_resumen_paciente():
    """Genera y descarga el PDF del resumen del paciente actual."""
    pac = Paciente.query.filter_by(usuario_id=current_user.id).first()
    if not pac:
        flash('No se encontró tu perfil de paciente.', 'danger')
        return redirect(url_for('paciente_dashboard'))

    seguimientos = pac.seguimientos.order_by(Seguimiento.fecha.desc()).limit(10).all()
    ultimo_seguimiento = seguimientos[0] if seguimientos else None

    pdf_buffer = generar_pdf_resumen_paciente(
        paciente=pac,
        seguimientos=seguimientos,
        ultimo_seguimiento=ultimo_seguimiento
    )

    filename = f"Mi_Resumen_Pie_{pac.nombre.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


@app.route('/medico/descargar_reporte_paciente/<int:paciente_id>')
@role_required('medico')
def descargar_reporte_paciente(paciente_id):
    """Descarga el PDF de la última evaluación de un paciente (para vista de evolución)."""
    paciente = db.session.get(Paciente, paciente_id)
    if not paciente:
        flash('Paciente no encontrado.', 'danger')
        return redirect(url_for('medico_dashboard'))

    # Buscar la última evaluación del médico actual para ese paciente
    evaluacion = Evaluacion.query.filter_by(
        paciente_id=paciente_id, medico_id=current_user.id
    ).order_by(Evaluacion.fecha.desc()).first()

    if not evaluacion:
        flash('No hay evaluaciones disponibles para este paciente.', 'danger')
        return redirect(url_for('medico_historial', paciente_id=paciente_id))

    medico = db.session.get(Usuario, current_user.id)

    pdf_buffer = generar_pdf_reporte_medico(
        evaluacion=evaluacion,
        paciente=paciente,
        medico=medico,
        upload_folder=app.config['UPLOAD_FOLDER']
    )

    filename = f"Reporte_{paciente.nombre.replace(' ', '_')}_{evaluacion.fecha.strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


# ── API endpoints para gráficas ────────────────────────────

@app.route('/api/evaluaciones/<int:paciente_id>')
@login_required
def api_evaluaciones(paciente_id):
    evaluaciones = Evaluacion.query.filter_by(paciente_id=paciente_id)\
                                    .order_by(Evaluacion.fecha.asc()).all()
    data = [{
        'fecha': e.fecha.strftime('%Y-%m-%d %H:%M'),
        'grado': e.grado,
        'gravedad_crisp': round(e.gravedad_crisp, 2) if e.gravedad_crisp else 0,
        'glucosa': e.glucosa_mgdl or 0
    } for e in evaluaciones]
    return jsonify(data)


# ── Inicialización de BD ────────────────────────────────────

def init_db():
    """Wrapper legacy — delega a auto_init_database()."""
    auto_init_database()


def auto_init_database():
    """
    Verifica e inicializa la base de datos automáticamente en el primer arranque.
    - Crea tablas si no existen.
    - Crea usuarios de prueba si la tabla está vacía.
    - NO borra datos existentes ni crea duplicados.
    """
    with app.app_context():
        try:
            print("\n✅ Verificando base de datos...")

            # 1. Crear tablas si no existen
            db.create_all()
            print("✅ Tablas verificadas / creadas correctamente")

            # 2. Verificar si ya hay usuarios
            usuarios_existentes = Usuario.query.count()
            if usuarios_existentes > 0:
                print(f"ℹ️  Base de datos ya inicializada ({usuarios_existentes} usuario(s) encontrados)")
                return

            # 3. No hay usuarios — crear los 3 de prueba
            print("⏳ No se encontraron usuarios. Creando usuarios de prueba...")

            # Admin
            admin = Usuario(username='admin', email='admin@piediabetico.mx',
                            nombre_completo='Administrador del Sistema', rol='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            print("   ✅ Usuario admin creado")

            # Médico
            medico = Usuario(username='dr_garcia', email='garcia@piediabetico.mx',
                             nombre_completo='Dr. Carlos García López', rol='medico',
                             cedula_profesional='12345678')
            medico.set_password('medico123')
            db.session.add(medico)
            print("   ✅ Usuario dr_garcia (médico) creado")

            # Paciente
            pac_user = Usuario(username='paciente1', email='paciente1@piediabetico.mx',
                               nombre_completo='María Hernández Ruiz', rol='paciente')
            pac_user.set_password('paciente123')
            db.session.add(pac_user)
            db.session.flush()
            print("   ✅ Usuario paciente1 creado")

            # Perfil de paciente asociado
            pac = Paciente(usuario_id=pac_user.id, nombre='María Hernández Ruiz',
                           edad=58, sexo='Femenino', tipo_diabetes='Tipo 2',
                           anios_diagnostico=12)
            db.session.add(pac)

            db.session.commit()
            print("✅ Usuarios de prueba creados exitosamente")
            print("✅ Base de datos inicializada correctamente\n")

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error al inicializar la base de datos: {e}")
            import traceback
            traceback.print_exc()


# ── Inicialización para producción (gunicorn) ───────────────

def inicializar_app():
    """Inicializa BD y sistema difuso. Llamado al arrancar."""
    init_db()
    with app.app_context():
        inicializar_sistema_difuso()
        # Modelo DL se carga lazy en la primera petición para ahorrar RAM
        if not IS_PRODUCTION:
            try:
                cargar_modelo_dl()
            except Exception as e:
                print(f"⚠️ Modelo DL no cargado al inicio: {e}")

    print("\n" + "=" * 60)
    print("🦶 SISTEMA INTEGRAL DE PIE DIABÉTICO — HeDiaF")
    print("   Flask Web Application v5.1")
    env_label = "PRODUCCIÓN (Render)" if IS_PRODUCTION else "DESARROLLO (local)"
    print(f"   Entorno: {env_label}")
    print("=" * 60 + "\n")


# Ejecutar inicialización al importar (para gunicorn)
inicializar_app()


# ── Main ────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n📋 Usuarios de prueba:")
    print("   Admin:    admin / admin123")
    print("   Médico:   dr_garcia / medico123")
    print("   Paciente: paciente1 / paciente123")
    print("=" * 60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000)