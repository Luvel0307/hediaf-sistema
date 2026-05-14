# -*- coding: utf-8 -*-
"""
Script de migración de base de datos.
Agrega los campos: edad, sexo, tipo_diabetes, anios_diagnostico
a la tabla 'pacientes' si no existen.
Compatible con bases de datos SQLite existentes.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'pie_diabetico.db')

def get_columns(cursor, table_name):
    """Obtiene los nombres de las columnas de una tabla."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"⚠️ Base de datos no encontrada en: {DB_PATH}")
        print("   La base de datos se creará automáticamente al iniciar la aplicación.")
        return

    print(f"📂 Conectando a: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        columns = get_columns(cursor, 'pacientes')
        print(f"   Columnas actuales en 'pacientes': {columns}")

        migrations = {
            'edad': 'ALTER TABLE pacientes ADD COLUMN edad INTEGER',
            'sexo': "ALTER TABLE pacientes ADD COLUMN sexo VARCHAR(20)",
            'tipo_diabetes': "ALTER TABLE pacientes ADD COLUMN tipo_diabetes VARCHAR(50)",
            'anios_diagnostico': 'ALTER TABLE pacientes ADD COLUMN anios_diagnostico INTEGER',
        }

        changes = 0
        for col_name, sql in migrations.items():
            if col_name not in columns:
                print(f"   ➕ Agregando columna: {col_name}")
                cursor.execute(sql)
                changes += 1
            else:
                print(f"   ✅ Columna ya existe: {col_name}")

        if changes > 0:
            conn.commit()
            print(f"\n✅ Migración completada: {changes} columna(s) agregada(s).")
            
            # Asignar valores por defecto a registros existentes sin datos
            cursor.execute("""
                UPDATE pacientes 
                SET edad = 0, sexo = 'No especificado', tipo_diabetes = 'No especificado'
                WHERE edad IS NULL AND sexo IS NULL AND tipo_diabetes IS NULL
            """)
            updated = cursor.rowcount
            if updated > 0:
                conn.commit()
                print(f"   📝 Se asignaron valores por defecto a {updated} registro(s) existente(s).")
        else:
            print("\n✅ La base de datos ya está actualizada. No se requieren cambios.")

    except Exception as e:
        print(f"\n❌ Error durante la migración: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == '__main__':
    print("=" * 50)
    print("🔄 MIGRACIÓN DE BASE DE DATOS")
    print("   Sistema Integral de Pie Diabético")
    print("=" * 50)
    migrate()
    print("=" * 50)
