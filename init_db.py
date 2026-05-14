# -*- coding: utf-8 -*-
"""
=============================================================
HeDiaF — Script de inicialización de base de datos
=============================================================
Crea todas las tablas y usuarios iniciales en PostgreSQL/SQLite.

Uso:
    python init_db.py              # Usa DATABASE_URL o SQLite por defecto
    DATABASE_URL=... python init_db.py  # Apunta a PostgreSQL
=============================================================
"""

import os
import sys

# Añadir directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Usuario, Paciente

def crear_tablas_y_usuarios():
    """Crea todas las tablas e inserta usuarios de demostración."""
    with app.app_context():
        print("📦 Creando tablas en la base de datos...")
        db.create_all()
        print("✅ Tablas creadas correctamente")

        cambios = False

        # --- Admin ---
        if not Usuario.query.filter_by(username='admin').first():
            admin = Usuario(
                username='admin',
                email='admin@piediabetico.mx',
                nombre_completo='Administrador del Sistema',
                rol='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            print("   ✅ Usuario admin creado")
            cambios = True

        # --- Médico demo ---
        if not Usuario.query.filter_by(username='dr_garcia').first():
            medico = Usuario(
                username='dr_garcia',
                email='garcia@piediabetico.mx',
                nombre_completo='Dr. Carlos García López',
                rol='medico',
                cedula_profesional='12345678'
            )
            medico.set_password('medico123')
            db.session.add(medico)
            print("   ✅ Usuario dr_garcia creado")
            cambios = True

        # --- Paciente demo ---
        if not Usuario.query.filter_by(username='paciente1').first():
            pac_user = Usuario(
                username='paciente1',
                email='paciente1@piediabetico.mx',
                nombre_completo='María Hernández Ruiz',
                rol='paciente'
            )
            pac_user.set_password('paciente123')
            db.session.add(pac_user)
            db.session.flush()

            pac = Paciente(
                usuario_id=pac_user.id,
                nombre='María Hernández Ruiz',
                edad=58,
                sexo='Femenino',
                tipo_diabetes='Tipo 2',
                anios_diagnostico=12
            )
            db.session.add(pac)
            print("   ✅ Usuario paciente1 y perfil Paciente creados")
            cambios = True

        if cambios:
            db.session.commit()
            print("\n✅ Base de datos inicializada correctamente")
        else:
            print("\nℹ️  Todos los usuarios ya existen, sin cambios")

        # Mostrar resumen
        total_usuarios = Usuario.query.count()
        total_pacientes = Paciente.query.count()
        print(f"\n📊 Resumen:")
        print(f"   Usuarios totales: {total_usuarios}")
        print(f"   Pacientes totales: {total_pacientes}")
        print(f"   BD: {app.config['SQLALCHEMY_DATABASE_URI'][:50]}...")


if __name__ == '__main__':
    print("=" * 60)
    print("🦶 HeDiaF — Inicialización de Base de Datos")
    print("=" * 60)
    crear_tablas_y_usuarios()
    print("\n" + "=" * 60)
    print("📋 Credenciales de prueba:")
    print("   Admin:    admin / admin123")
    print("   Médico:   dr_garcia / medico123")
    print("   Paciente: paciente1 / paciente123")
    print("=" * 60)
