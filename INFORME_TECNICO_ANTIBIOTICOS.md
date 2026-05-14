# Informe Técnico: Protocolo de Asignación de Antibióticos en el Sistema Integral de Pie Diabético

**Versión:** 1.0  
**Fecha:** 22 de marzo de 2026  
**Autora:** Guadalupe Vélez Pérez  
**Sistema:** Sistema Integral para Evaluación y Seguimiento del Pie Diabético v5.0  

---

## 1. Introducción

### 1.1 Objetivo del sistema de recomendación

El presente informe describe el protocolo algorítmico mediante el cual el **Sistema Integral de Pie Diabético** genera recomendaciones de antibióticos para infecciones del pie diabético. El sistema emplea un enfoque híbrido que combina **Lógica Difusa Tipo-2 Intervalar (IT2-FLS)** con **Deep Learning (MobileNetV2)** para determinar el nivel de gravedad de la infección y, a partir de este, sugerir un esquema antibiótico basado en evidencia clínica.

El objetivo primordial es proporcionar al profesional de salud una **herramienta de apoyo a la decisión clínica** que sistematice la evaluación de gravedad y ofrezca recomendaciones terapéuticas concordantes con las guías internacionales más recientes.

### 1.2 Importancia de la selección adecuada de antibióticos

Las infecciones del pie diabético (IPD) representan una de las complicaciones más graves de la diabetes mellitus. Según la Organización Mundial de la Salud y la Federación Internacional de Diabetes:

- Aproximadamente el **25% de los pacientes con diabetes** desarrollarán una úlcera en el pie a lo largo de su vida (Armstrong et al., 2017).
- Las infecciones del pie diabético son la causa más frecuente de **hospitalización** relacionada con diabetes en países en desarrollo (Lavery et al., 2006).
- Una selección inadecuada de antibióticos puede derivar en **resistencia antimicrobiana**, progresión de la infección, sepsis y **amputación** (Lipsky et al., 2020).
- En México, la diabetes es la primera causa de amputación no traumática de miembros inferiores, con más de **75,000 amputaciones anuales** (ENSANUT, 2018).

La selección empírica inicial de antibióticos debe guiarse por la **gravedad de la infección**, los **patógenos más probables** según el nivel de severidad, y las **guías clínicas basadas en evidencia** (IWGDF, 2023; IDSA, 2012).

---

## 2. Marco Teórico

### 2.1 Guía IWGDF 2023 (International Working Group on the Diabetic Foot)

La guía IWGDF 2023 establece el estándar internacional para la clasificación y manejo de infecciones del pie diabético. Sus principales aportaciones incluyen:

- **Sistema de clasificación IWGDF/IDSA** que estratifica las infecciones en cuatro niveles: no infectado, infección leve, infección moderada e infección grave.
- **Criterios diagnósticos** basados en signos clínicos locales (calor, eritema, dolor/sensibilidad, tumefacción, induración) y sistémicos (fiebre, taquicardia, leucocitosis).
- **Protocolos de tratamiento antibiótico empírico** diferenciados según la gravedad.
- Recomendación de **cobertura para cocos grampositivos** (especialmente *Staphylococcus aureus*) en infecciones leves, y **cobertura ampliada** incluyendo gramnegativos y anaerobios en infecciones moderadas a graves (Senneville et al., 2024).

### 2.2 NOM-015-SSA2-2010 (Norma Oficial Mexicana)

La Norma Oficial Mexicana NOM-015-SSA2-2010 para la prevención, tratamiento y control de la diabetes mellitus establece:

- La obligatoriedad de la **exploración periódica del pie** en todo paciente diabético.
- Criterios para la **clasificación del riesgo** de pie diabético.
- Lineamientos para el **tratamiento integral** que incluye control glucémico, cuidado local de la herida y terapia antimicrobiana.
- La necesidad de **referencia oportuna** a segundo o tercer nivel de atención cuando la gravedad lo amerite (Secretaría de Salud, 2010).

### 2.3 Clasificación de infecciones en pie diabético

El sistema implementado utiliza una clasificación de cuatro niveles, alineada con IWGDF/IDSA:

| Grado | Clasificación | Criterios Clínicos | Acción Requerida |
|:-----:|:--------------|:-------------------|:-----------------|
| **1** | No infectado | Ausencia de signos de infección; úlcera limpia o herida en proceso de cicatrización | Control y vigilancia rutinaria |
| **2** | Infección leve | ≥2 signos locales de inflamación; eritema <2 cm alrededor de la úlcera; infección limitada a piel y tejido subcutáneo superficial; sin signos sistémicos | Consulta médica en 24-48 horas |
| **3** | Infección moderada | Eritema ≥2 cm; infección profunda (fascia, músculo, tendón, hueso); absceso; gangrena localizada; sin respuesta inflamatoria sistémica | Valoración médica el mismo día |
| **4** | Infección grave | Cualquier infección del pie con signos de síndrome de respuesta inflamatoria sistémica (SRIS): fiebre >38°C o <36°C, FC >90 lpm, FR >20 rpm, leucocitos >12,000 o <4,000; isquemia crítica; sepsis | Traslado inmediato a urgencias |

*Tabla 1. Clasificación de gravedad de infecciones del pie diabético. Adaptado de IWGDF 2023 e IDSA 2012.*

### 2.4 Guía IDSA 2012 para Infecciones del Pie Diabético

La Infectious Diseases Society of America (IDSA) publicó en 2012 la guía de práctica clínica para el diagnóstico y tratamiento de infecciones del pie diabético, la cual complementa las recomendaciones de IWGDF:

- Define los **criterios diagnósticos** de infección basados en hallazgos clínicos.
- Establece **algoritmos de tratamiento empírico** según la gravedad.
- Recomienda **cultivos** de tejido profundo (no superficiales) para guiar el tratamiento definitivo.
- Proporciona **esquemas antibióticos específicos** para cada nivel de gravedad (Lipsky et al., 2012).

---

## 3. Metodología de Evaluación

### 3.1 Sistema de Lógica Difusa Tipo-2 Intervalar (IT2-FLS)

El sistema emplea un **Sistema de Lógica Difusa Tipo-2 Intervalar** (Interval Type-2 Fuzzy Logic System, IT2-FLS), que constituye una extensión de los sistemas difusos tipo-1 convencionales. La principal ventaja del IT2-FLS radica en su capacidad para modelar la **incertidumbre inherente** a las variables clínicas mediante la denominada **Huella de Incertidumbre** (Footprint of Uncertainty, FOU).

#### 3.1.1 Arquitectura del IT2-FLS

El sistema implementa dos motores de inferencia paralelos:

```
                    ┌──────────────────────┐
                    │  Entrada Clínica     │
                    │  (6 variables)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Fuzzificación      │
                    │   Tipo-2 Intervalar  │
                    └──────────┬───────────┘
                       ┌───────┴────────┐
                       │                │
              ┌────────▼──────┐ ┌───────▼───────┐
              │ Motor Inferior│ │ Motor Superior│
              │   (Lower MF)  │ │   (Upper MF)  │
              └────────┬──────┘ └───────┬───────┘
                       │                │
              ┌────────▼──────┐ ┌───────▼───────┐
              │  Inferencia   │ │  Inferencia   │
              │  + Defuzz.    │ │  + Defuzz.    │
              │  (Centroide)  │ │  (Centroide)  │
              └────────┬──────┘ └───────┬───────┘
                       │                │
                       │   rL      rU   │
                       └───────┬────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Reducción de Tipo   │
                    │  rC = (rL + rU) / 2  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Clasificación +     │
                    │  Nivel de Confianza  │
                    └──────────────────────┘
```

*Figura 1. Arquitectura del Sistema de Lógica Difusa Tipo-2 Intervalar.*

- **rL**: Resultado del motor inferior (Lower Membership Function).
- **rU**: Resultado del motor superior (Upper Membership Function).
- **rC**: Valor crisp final = (rL + rU) / 2.
- **Incertidumbre**: |rU - rL|.
- **Confianza**: ALTA si incertidumbre < 0.3; MEDIA si < 0.6; BAJA en otro caso.

### 3.2 Variables clínicas evaluadas (6 variables de entrada)

El sistema evalúa seis variables clínicas de entrada, seleccionadas con base en los criterios diagnósticos de IWGDF 2023 e IDSA 2012:

| Variable | Nombre Interno | Universo de Discurso | Conjuntos Difusos | Descripción Clínica |
|:---------|:---------------|:---------------------|:-------------------|:-------------------|
| **Signos locales** | `SignosLocales` | [0, 4] | ninguno, pocos, varios, muchos | Número de signos locales de infección presentes (calor, eritema, dolor/sensibilidad, tumefacción, induración). Escala de 0 a 4 signos. |
| **Eritema (cm)** | `EritemaCm` | [0, 5] | minimo, pequeño, grande | Extensión del eritema perilesional medida en centímetros desde el borde de la úlcera. El umbral de 2 cm distingue infección leve de moderada (IWGDF). |
| **Profundidad** | `Profundidad` | [0, 2] | sin_herida, superficial, profunda | Nivel de compromiso tisular: 0 = sin herida abierta; 1 = superficial (piel/subcutáneo); 2 = profunda (fascia/músculo/hueso). |
| **Signos sistémicos** | `SignosSist` | [0, 1] | no, si | Presencia de signos de respuesta inflamatoria sistémica: fiebre >38°C, taquicardia >90 lpm, taquipnea >20 rpm, leucocitosis. 0 = ausentes; 1 = presentes. |
| **Isquemia** | `Isquemia` | [0, 2] | ninguna, leve, alta | Estado de perfusión vascular del miembro: 0 = sin isquemia; 1 = isquemia leve (pulsos disminuidos, pie frío); 2 = isquemia crítica (ausencia de pulsos, cianosis). |
| **Glucosa (mg/dL)** | `GlucosaMgdl` | [70, 500] | normal, elevada, muy_elevada | Nivel de glucemia capilar en mg/dL. El descontrol glucémico >250 mg/dL se considera un factor agravante de la infección. |

*Tabla 2. Variables clínicas de entrada del sistema difuso.*

### 3.3 Funciones de membresía

Cada variable posee funciones de membresía tanto para el límite inferior (Lower MF) como para el límite superior (Upper MF), creando la Huella de Incertidumbre. Se utilizan funciones triangulares (`trimf`) y trapezoidales (`trapmf`).

#### 3.3.1 Signos Locales (`SignosLocales`)

| Conjunto | Lower MF (trimf) | Upper MF (trimf) |
|:---------|:------------------|:------------------|
| ninguno  | [0, 0, 0.8]      | [0, 0, 1.2]      |
| pocos    | [0, 1, 1.8]      | [0, 1, 2.2]      |
| varios   | [1.8, 3, 3.8]    | [1.5, 3, 4]      |
| muchos   | [3.2, 4, 4]      | [2.8, 4, 4]      |

#### 3.3.2 Eritema (`EritemaCm`)

| Conjunto | Lower MF          | Upper MF           |
|:---------|:-------------------|:-------------------|
| minimo   | trapmf [0, 0, 0.2, 0.4] | trapmf [0, 0, 0.4, 0.6] |
| pequeño  | trimf [0.3, 1.0, 1.8]   | trimf [0.2, 1.0, 2.2]   |
| grande   | trapmf [1.9, 2.2, 5, 5] | trapmf [1.6, 1.9, 5, 5] |

#### 3.3.3 Profundidad (`Profundidad`)

| Conjunto    | Lower MF (d=0.4)     | Upper MF (d=0.6)     |
|:------------|:----------------------|:----------------------|
| sin_herida  | trimf [0, 0, 0.4]    | trimf [0, 0, 0.6]    |
| superficial | trimf [0.6, 1, 1.4]  | trimf [0.4, 1, 1.6]  |
| profunda    | trimf [1.6, 2, 2]    | trimf [1.4, 2, 2]    |

#### 3.3.4 Signos Sistémicos (`SignosSist`)

| Conjunto | Lower MF (d=0.4)  | Upper MF (d=0.6)  |
|:---------|:-------------------|:-------------------|
| no       | trimf [0, 0, 0.4] | trimf [0, 0, 0.6] |
| si       | trimf [0.6, 1, 1] | trimf [0.4, 1, 1] |

#### 3.3.5 Isquemia (`Isquemia`)

| Conjunto | Lower MF (d=0.4)     | Upper MF (d=0.6)     |
|:---------|:----------------------|:----------------------|
| ninguna  | trimf [0, 0, 0.4]    | trimf [0, 0, 0.6]    |
| leve     | trimf [0.6, 1, 1.4]  | trimf [0.4, 1, 1.6]  |
| alta     | trimf [1.6, 2, 2]    | trimf [1.4, 2, 2]    |

#### 3.3.6 Glucosa (`GlucosaMgdl`)

| Conjunto     | Lower MF                  | Upper MF                  |
|:-------------|:--------------------------|:--------------------------|
| normal       | trapmf [70, 70, 110, 130] | trapmf [70, 70, 120, 145] |
| elevada      | trimf [120, 180, 250]     | trimf [110, 180, 265]     |
| muy_elevada  | trapmf [240, 270, 500, 500] | trapmf [225, 255, 500, 500] |

#### 3.3.7 Variable de salida: Gravedad

| Conjunto  | Lower MF (trimf)  | Upper MF (trimf)  |
|:----------|:-------------------|:-------------------|
| baja      | [1, 1, 1.8]       | [1, 1, 2.0]       |
| leve      | [1.6, 2, 2.4]     | [1.5, 2, 2.6]     |
| moderada  | [2.3, 3, 3.4]     | [2.2, 3, 3.6]     |
| grave     | [3.3, 4, 4]       | [3.2, 4, 4]       |

*Tablas 3-9. Funciones de membresía para cada variable del sistema IT2-FLS.*

### 3.4 Reglas de inferencia

El sistema emplea **13 reglas de inferencia difusa** tipo Mamdani, diseñadas con base en los criterios diagnósticos de IWGDF 2023 e IDSA 2012. Las mismas reglas se aplican tanto al motor inferior como al superior:

#### Reglas para gravedad BAJA (Grado 1):
| # | Regla |
|:-:|:------|
| R1 | **SI** SignosLocales=ninguno **Y** SignosSist=no **Y** Glucosa=normal **ENTONCES** Gravedad=baja |
| R2 | **SI** SignosLocales=pocos **Y** Eritema=minimo **Y** Profundidad=sin_herida **Y** SignosSist=no **ENTONCES** Gravedad=baja |

#### Reglas para gravedad LEVE (Grado 2):
| # | Regla |
|:-:|:------|
| R3 | **SI** SignosLocales=varios **Y** Eritema=pequeño **Y** Profundidad=superficial **Y** SignosSist=no **Y** Isquemia=ninguna **ENTONCES** Gravedad=leve |
| R4 | **SI** SignosLocales=pocos **Y** Eritema=pequeño **Y** Profundidad=superficial **Y** SignosSist=no **ENTONCES** Gravedad=leve |

#### Reglas para gravedad MODERADA (Grado 3):
| # | Regla |
|:-:|:------|
| R5 | **SI** Eritema=grande **Y** SignosSist=no **ENTONCES** Gravedad=moderada |
| R6 | **SI** Profundidad=profunda **Y** SignosSist=no **Y** SignosLocales=varios **ENTONCES** Gravedad=moderada |
| R7 | **SI** SignosLocales=muchos **Y** Eritema=pequeño **Y** SignosSist=no **ENTONCES** Gravedad=moderada |
| R8 | **SI** Glucosa=muy_elevada **Y** SignosLocales=varios **Y** SignosSist=no **ENTONCES** Gravedad=moderada |

#### Reglas para gravedad GRAVE (Grado 4):
| # | Regla |
|:-:|:------|
| R9 | **SI** SignosSist=si **O** Isquemia=alta **ENTONCES** Gravedad=grave |
| R10 | **SI** Isquemia=alta **Y** Profundidad=profunda **ENTONCES** Gravedad=grave |
| R11 | **SI** SignosSist=si **Y** Eritema=grande **ENTONCES** Gravedad=grave |
| R12 | **SI** Glucosa=muy_elevada **Y** SignosSist=si **ENTONCES** Gravedad=grave |
| R13 | **SI** Glucosa=muy_elevada **Y** Isquemia=alta **ENTONCES** Gravedad=grave |

*Tabla 10. Reglas de inferencia del sistema difuso.*

**Justificación clínica de las reglas:**
- **R1-R2**: Concordancia con criterio IWGDF de "no infectado": ausencia de signos clínicos de infección.
- **R3-R4**: Concordancia con criterio de infección leve IWGDF: signos locales presentes, eritema <2 cm, sin compromiso profundo ni sistémico.
- **R5-R8**: Criterios de infección moderada: eritema ≥2 cm (R5), compromiso profundo (R6), múltiples signos locales (R7), descontrol glucémico grave con signos locales (R8).
- **R9-R13**: Criterios de infección grave: presencia de signos sistémicos o isquemia crítica, concordante con riesgo de sepsis y pérdida de extremidad.

### 3.5 Cálculo del nivel de gravedad

El proceso de cálculo sigue estos pasos:

1. **Fuzzificación**: Las 6 variables clínicas numéricas se convierten en grados de pertenencia usando las funciones de membresía Lower y Upper.
2. **Inferencia** (× 2 motores): Cada motor aplica las 13 reglas sobre sus respectivas funciones de membresía.
3. **Defuzzificación**: Se utiliza el método del **centroide** para obtener un valor numérico de cada motor.
4. **Reducción de tipo**: Se calcula el valor crisp final como promedio de ambos resultados:
   - `rC = (rL + rU) / 2`
5. **Clasificación**: El valor crisp se combina con criterios clínicos determinísticos para asignar el grado final (1-4).

#### Criterios determinísticos de clasificación (post-difuso):

```
SI  signos_sistémicos == 1  O  isquemia == 2  O  rC ≥ 3.5
    → Grado 4 (Grave)

SI  eritema ≥ 2.0  O  profundidad == 2  O
    (signos_locales ≥ 3 Y eritema ≥ 2.0)  O
    (glucosa > 250 Y signos_locales ≥ 2 Y signos_sistémicos == 0)
    → Grado 3 (Moderada)

SI  signos_locales ≥ 2  Y  eritema < 2.0  Y  profundidad < 2  Y  signos_sistémicos == 0
    → Grado 2 (Leve)

EN OTRO CASO
    → Grado 1 (No infectado)
```

Este mecanismo híbrido (difuso + determinístico) garantiza que condiciones clínicas críticas como la presencia de **signos sistémicos** o **isquemia crítica** siempre generen una clasificación de gravedad máxima, independientemente del resultado numérico difuso.

---

## 4. Protocolo de Asignación de Antibióticos

### 4.1 Tabla de asignación: Nivel de gravedad → Antibiótico recomendado

| Grado | Clasificación | Esquema de Primera Línea | Esquema Alternativo | Vía | Duración |
|:-----:|:--------------|:------------------------|:--------------------|:---:|:---------|
| **1** | No infectado | **No requiere antibióticos** | N/A | N/A | N/A |
| **2** | Infección leve | **Cefalexina 500 mg c/6h** | **Amoxicilina-clavulánico 875/125 mg c/12h** | VO | 7-14 días |
| **3** | Infección moderada | **Amoxicilina-clavulánico 1 g c/8h** | **Clindamicina 600 mg c/8h + Ciprofloxacino 400 mg c/12h** | IV o VO | 14-21 días |
| **4** | Infección grave | **Piperacilina-tazobactam 4.5 g c/6h** | **Meropenem 1 g c/8h + Vancomicina 15-20 mg/kg c/12h** | IV | 21-28 días* |

*Tabla 11. Protocolo de asignación de antibióticos según nivel de gravedad.*

> \* La duración en infecciones graves depende de la respuesta clínica, resultados de cultivos, y presencia de osteomielitis. Requiere ajuste por infectología.

### 4.2 Justificación clínica de cada recomendación

#### Grado 1 — No infectado: Sin antibióticos

**Justificación:** Las guías IWGDF 2023 e IDSA 2012 son enfáticas en que las úlceras no infectadas **no deben recibir tratamiento antibiótico**, ni sistémico ni tópico, ya que:
- No acelera la cicatrización.
- Promueve la selección de bacterias resistentes.
- Genera efectos adversos innecesarios.

El manejo se centra en: limpieza con solución salina, apósitos estériles, descarga de presión, y control glucémico (Senneville et al., 2024).

#### Grado 2 — Infección leve: Cefalexina o Amoxicilina-clavulánico

**Primera línea: Cefalexina 500 mg c/6h VO**
- **Espectro de cobertura**: Cefalosporina de primera generación con excelente actividad contra cocos grampositivos, especialmente *Staphylococcus aureus* meticilino-sensible (MSSA) y *Streptococcus* spp.
- **Justificación**: Las infecciones leves son predominantemente causadas por **cocos grampositivos aerobios** (IWGDF, 2023). *S. aureus* es el patógeno más frecuente (>50% de los aislamientos). La cefalexina proporciona cobertura adecuada con buena biodisponibilidad oral y perfil de seguridad favorable.
- **Evidencia**: Recomendación concordante con IDSA 2012 (Lipsky et al., 2012) e IWGDF 2023 (Senneville et al., 2024).

**Alternativa: Amoxicilina-clavulánico 875/125 mg c/12h VO**
- **Espectro de cobertura**: Aminopenicilina con inhibidor de betalactamasa. Cobertura ampliada hacia grampositivos, gramnegativos y **anaerobios**.
- **Justificación**: Indicada cuando se sospecha flora polimicrobiana o hay alergia a cefalosporinas sin anafilaxia cruzada. Proporciona cobertura para *S. aureus*, *Streptococcus*, *E. coli*, *Proteus*, y anaerobios como *Bacteroides* spp.

#### Grado 3 — Infección moderada: Amoxicilina-clavulánico IV o Clindamicina + Ciprofloxacino

**Primera línea: Amoxicilina-clavulánico 1 g c/8h IV o VO**
- **Espectro de cobertura**: Amplio espectro que incluye grampositivos, gramnegativos y anaerobios.
- **Justificación**: Las infecciones moderadas frecuentemente involucran **flora polimicrobiana** que incluye aerobios y anaerobios. La amoxicilina-clavulánico a dosis más altas y por vía intravenosa (con posibilidad de cambio secuencial a vía oral) proporciona cobertura adecuada para la mayoría de patógenos esperados.
- **Consideración**: Se puede iniciar IV y pasar a VO cuando haya mejoría clínica (terapia secuencial).

**Alternativa: Clindamicina 600 mg c/8h + Ciprofloxacino 400 mg c/12h IV**
- **Espectro combinado**:
  - *Clindamicina*: Excelente cobertura para grampositivos (incluyendo MRSA comunitario en algunas regiones), anaerobios, y tiene efecto antitoxina.
  - *Ciprofloxacino*: Fluoroquinolona con cobertura para gramnegativos, incluyendo *Pseudomonas aeruginosa*.
- **Justificación**: Esta combinación proporciona cobertura empírica amplia y es útil en pacientes con alergia a betalactámicos. La sinergia cubre el espectro polimicrobiano típico de infecciones moderadas (Lipsky et al., 2012).

#### Grado 4 — Infección grave: Piperacilina-tazobactam o Meropenem + Vancomicina

**Primera línea: Piperacilina-tazobactam 4.5 g c/6h IV**
- **Espectro de cobertura**: Ureidopenicilina con inhibidor de betalactamasa. Uno de los espectros más amplios disponibles: grampositivos, gramnegativos (incluyendo *Pseudomonas aeruginosa*), y anaerobios.
- **Justificación**: Las infecciones graves con riesgo de sepsis requieren **cobertura empírica de amplio espectro** inmediata. Piperacilina-tazobactam es la primera línea recomendada por IWGDF 2023 para infecciones graves que amenazan la vida o la extremidad. Cubre la flora polimicrobiana habitual incluyendo *Pseudomonas* (Senneville et al., 2024).

**Alternativa: Meropenem 1 g c/8h IV + Vancomicina 15-20 mg/kg c/12h IV**
- **Espectro combinado**:
  - *Meropenem*: Carbapenémico de amplio espectro. Cobertura para gramnegativos multirresistentes, incluyendo productores de BLEE y *Pseudomonas*. Anaerobios.
  - *Vancomicina*: Glucopéptido con actividad específica contra grampositivos resistentes, especialmente **MRSA** (*Staphylococcus aureus* meticilino-resistente) y *Enterococcus* spp.
- **Justificación**: Esta combinación es de **rescate** y se indica cuando:
  - Hay sospecha de MRSA (infección previa, hospitalización reciente, alta prevalencia local).
  - Falla terapéutica con piperacilina-tazobactam.
  - Infecciones en pacientes con uso previo de antibióticos de amplio espectro.
  - Riesgo de organismos multirresistentes (IWGDF, 2023).
- **Dosis de vancomicina**: 15-20 mg/kg basado en peso corporal real, requiere monitorización de niveles séricos.

### 4.3 Espectro de cobertura comparativo

| Antibiótico | Grampositivos | Gramnegativos | Anaerobios | MRSA | *Pseudomonas* |
|:------------|:-------------:|:-------------:|:----------:|:----:|:--------------:|
| Cefalexina | ✅✅ | ❌ | ❌ | ❌ | ❌ |
| Amoxicilina-clavulánico | ✅✅ | ✅ | ✅ | ❌ | ❌ |
| Clindamicina | ✅ | ❌ | ✅✅ | ✅* | ❌ |
| Ciprofloxacino | ✅ | ✅✅ | ❌ | ❌ | ✅ |
| Piperacilina-tazobactam | ✅✅ | ✅✅ | ✅✅ | ❌ | ✅✅ |
| Meropenem | ✅✅ | ✅✅✅ | ✅✅ | ❌ | ✅✅ |
| Vancomicina | ✅✅✅ | ❌ | ❌ | ✅✅✅ | ❌ |

*Tabla 12. Espectro de cobertura antimicrobiana comparativo.*

> ✅✅✅ = Excelente cobertura; ✅✅ = Buena cobertura; ✅ = Cobertura parcial; ❌ = Sin cobertura.  
> \* La actividad de clindamicina contra MRSA varía según la región y epidemiología local.

---

## 5. Algoritmo de Decisión

### 5.1 Diagrama de flujo (pseudocódigo)

```
ALGORITMO: Asignación de Antibióticos en Pie Diabético
ENTRADA: signos_locales, eritema_cm, profundidad, signos_sist, isquemia, glucosa_mgdl
SALIDA: grado, etiqueta, antibiótico_recomendado

INICIO
│
├── 1. VALIDAR ENTRADAS
│   └── Clipear valores a rangos válidos
│       signos_locales ← clip(signos_locales, 0, 4)
│       eritema_cm     ← clip(eritema_cm, 0, 5)
│       profundidad    ← clip(profundidad, 0, 2)
│       signos_sist    ← clip(signos_sist, 0, 1)
│       isquemia       ← clip(isquemia, 0, 2)
│       glucosa_mgdl   ← clip(glucosa_mgdl, 70, 500)
│
├── 2. EVALUACIÓN DIFUSA TIPO-2
│   ├── Ejecutar motor LOWER → rL ← clip(resultado, 1.0, 4.0)
│   ├── Ejecutar motor UPPER → rU ← clip(resultado, 1.0, 4.0)
│   ├── rC ← (rL + rU) / 2
│   └── confianza ← ALTA si |rU-rL| < 0.3, MEDIA si < 0.6, BAJA si ≥ 0.6
│
├── 3. CLASIFICACIÓN HÍBRIDA (Difuso + Determinístico)
│   │
│   ├── ¿signos_sist == 1 OR isquemia == 2 OR rC ≥ 3.5?
│   │   └── SÍ → grado = 4 (GRAVE)
│   │         antibiótico = "Piperacilina-tazobactam 4.5 g c/6h IV"
│   │         alternativa = "Meropenem 1 g c/8h IV + Vancomicina 15-20 mg/kg c/12h IV"
│   │
│   ├── ¿eritema ≥ 2.0 OR profundidad == 2 OR
│   │   (signos_locales ≥ 3 AND eritema ≥ 2.0) OR
│   │   (glucosa > 250 AND signos_locales ≥ 2 AND signos_sist == 0)?
│   │   └── SÍ → grado = 3 (MODERADA)
│   │         antibiótico = "Amoxicilina-clavulánico 1 g c/8h IV o VO"
│   │         alternativa = "Clindamicina 600 mg c/8h + Ciprofloxacino 400 mg c/12h IV"
│   │
│   ├── ¿signos_locales ≥ 2 AND eritema < 2.0 AND profundidad < 2 AND signos_sist == 0?
│   │   └── SÍ → grado = 2 (LEVE)
│   │         antibiótico = "Cefalexina 500 mg c/6h VO × 7-14 días"
│   │         alternativa = "Amoxicilina-clavulánico 875/125 mg c/12h VO"
│   │
│   └── EN OTRO CASO
│         → grado = 1 (NO INFECTADO)
│           antibiótico = "No requiere antibióticos"
│
├── 4. GENERAR ALERTA (si grado ≥ 3)
│   └── Crear alerta automática para el médico asignado
│
└── 5. RETORNAR RESULTADO
    └── {grado, etiqueta, antibiótico, alternativa, rC, confianza, recomendaciones}

FIN
```

*Figura 2. Pseudocódigo del algoritmo de decisión para asignación de antibióticos.*

### 5.2 Ejemplos de casos clínicos

#### Caso 1: Úlcera no infectada (Grado 1)

| Parámetro | Valor |
|:----------|:------|
| Signos locales | 0 |
| Eritema | 0 cm |
| Profundidad | 0 (sin herida) |
| Signos sistémicos | No |
| Isquemia | Ninguna |
| Glucosa | 110 mg/dL |

**Resultado esperado:**
- Grado: 1 — No infectado
- Antibiótico: No requiere antibióticos
- Recomendación: Control y vigilancia rutinaria. Limpieza diaria con solución salina, apósito estéril.

---

#### Caso 2: Infección leve (Grado 2)

| Parámetro | Valor |
|:----------|:------|
| Signos locales | 2 (eritema + calor) |
| Eritema | 1.5 cm |
| Profundidad | 1 (superficial) |
| Signos sistémicos | No |
| Isquemia | Ninguna |
| Glucosa | 160 mg/dL |

**Resultado esperado:**
- Grado: 2 — Infección leve
- Antibiótico: Cefalexina 500 mg c/6h VO × 7-14 días
- Alternativa: Amoxicilina-clavulánico 875/125 mg c/12h VO
- Recomendación: Consulta médica en 24-48 horas. Limpieza diaria.

---

#### Caso 3: Infección moderada (Grado 3)

| Parámetro | Valor |
|:----------|:------|
| Signos locales | 3 (eritema + calor + tumefacción) |
| Eritema | 3.0 cm |
| Profundidad | 2 (profunda) |
| Signos sistémicos | No |
| Isquemia | Leve |
| Glucosa | 280 mg/dL |

**Resultado esperado:**
- Grado: 3 — Infección moderada
- Antibiótico: Amoxicilina-clavulánico 1 g c/8h IV o VO
- Alternativa: Clindamicina 600 mg c/8h + Ciprofloxacino 400 mg c/12h IV
- Recomendación: Valoración médica hoy mismo. Posible hospitalización.

---

#### Caso 4: Infección grave (Grado 4)

| Parámetro | Valor |
|:----------|:------|
| Signos locales | 4 (todos presentes) |
| Eritema | 4.5 cm |
| Profundidad | 2 (profunda) |
| Signos sistémicos | Sí (fiebre 39°C) |
| Isquemia | Alta (crítica) |
| Glucosa | 350 mg/dL |

**Resultado esperado:**
- Grado: 4 — Infección grave
- Antibiótico: Piperacilina-tazobactam 4.5 g c/6h IV
- Alternativa: Meropenem 1 g c/8h IV + Vancomicina 15-20 mg/kg c/12h IV
- Recomendación: TRASLADO INMEDIATO A URGENCIAS. Riesgo de sepsis y amputación. Hospitalización urgente.

---

## 6. Validación Clínica

### 6.1 Concordancia con guías internacionales

El protocolo implementado fue diseñado para mantener concordancia con las siguientes guías:

| Aspecto | IWGDF 2023 | IDSA 2012 | NOM-015-SSA2 | Sistema Implementado |
|:--------|:----------:|:---------:|:------------:|:-------------------:|
| Clasificación 4 niveles | ✅ | ✅ | — | ✅ |
| No antibiótico si no infectado | ✅ | ✅ | — | ✅ |
| Cefalosporinas 1ª gen. para leve | ✅ | ✅ | — | ✅ |
| Cobertura polimicrobiana en moderada | ✅ | ✅ | — | ✅ |
| Amplio espectro IV en grave | ✅ | ✅ | — | ✅ |
| Cobertura anti-MRSA en grave | ✅ | ✅ | — | ✅ |
| Referencia oportuna | — | — | ✅ | ✅ |
| Control glucémico integral | — | — | ✅ | ✅ |

*Tabla 13. Concordancia del sistema con guías clínicas internacionales y nacionales.*

### 6.2 Casos de uso apropiados

El sistema está diseñado para ser utilizado en:

1. **Consulta de primer contacto**: Apoyo al médico general en la evaluación inicial de úlceras del pie diabético.
2. **Evaluación en consultorios de medicina familiar**: Herramienta para determinar la urgencia de referencia.
3. **Capacitación y educación médica**: Material didáctico para enseñanza de protocolos de manejo de pie diabético.
4. **Investigación clínica**: Marco de referencia para estudios sobre tratamiento de infecciones del pie diabético.
5. **Seguimiento ambulatorio**: Monitoreo longitudinal de la evolución del paciente con autoevaluaciones.

### 6.3 Limitaciones del sistema

Es fundamental reconocer las siguientes limitaciones:

1. **No sustituye el juicio clínico**: El sistema es una herramienta de apoyo. La decisión final siempre corresponde al profesional de salud.
2. **No considera cultivos microbiológicos**: Las recomendaciones son empíricas. El tratamiento definitivo debe guiarse por resultados de cultivo y antibiograma.
3. **No ajusta por epidemiología local**: Los patrones de resistencia antimicrobiana varían según la región geográfica e institución. Los esquemas deben ajustarse según antibiogramas locales.
4. **No evalúa alergias medicamentosas**: El sistema no pregunta ni considera alergias del paciente a antibióticos.
5. **No ajusta por función renal/hepática**: Las dosis sugeridas son estándar y requieren ajuste según función orgánica del paciente.
6. **No considera interacciones medicamentosas**: El sistema no evalúa posibles interacciones con otros medicamentos que el paciente esté tomando.
7. **No diagnostica osteomielitis**: La sospecha de osteomielitis requiere estudios complementarios (radiografía, RMN, biopsia ósea) que están fuera del alcance del sistema.
8. **Modelo de imagen limitado**: El modelo de Deep Learning (MobileNetV2) fue entrenado con un dataset específico y su precisión puede variar en poblaciones diferentes a la de entrenamiento.

---

## 7. Referencias Bibliográficas

1. Armstrong, D. G., Boulton, A. J. M., & Bus, S. A. (2017). Diabetic Foot Ulcers and Their Recurrence. *New England Journal of Medicine*, 376(24), 2367–2375. https://doi.org/10.1056/NEJMra1615439

2. Bus, S. A., Lavery, L. A., Monteiro‐Soares, M., Rasmussen, A., Raspovic, A., Sacco, I. C., & van Netten, J. J. (2020). Guidelines on the prevention of foot ulcers in persons with diabetes (IWGDF 2019 update). *Diabetes/Metabolism Research and Reviews*, 36(S1), e3269. https://doi.org/10.1002/dmrr.3269

3. Castillo, O., & Melin, P. (2008). *Type-2 Fuzzy Logic: Theory and Applications*. Studies in Fuzziness and Soft Computing, Vol. 223. Springer. https://doi.org/10.1007/978-3-540-76284-3

4. Encuesta Nacional de Salud y Nutrición (ENSANUT). (2018). *Resultados nacionales 2018*. Instituto Nacional de Salud Pública. México.

5. International Working Group on the Diabetic Foot (IWGDF). (2023). *IWGDF Guidelines on the diagnosis and treatment of diabetes-related foot infections (2023 update)*. https://iwgdfguidelines.org/

6. Lavery, L. A., Armstrong, D. G., Wunderlich, R. P., Mohler, M. J., Wendel, C. S., & Lipsky, B. A. (2006). Risk factors for foot infections in individuals with diabetes. *Diabetes Care*, 29(6), 1288–1293. https://doi.org/10.2337/dc05-2425

7. Lipsky, B. A., Aragón-Sánchez, J., Diggle, M., Embil, J., Kono, S., Lavery, L., ... & Peters, E. J. (2016). IWGDF guidance on the diagnosis and management of foot infections in persons with diabetes. *Diabetes/Metabolism Research and Reviews*, 32(S1), 45–74. https://doi.org/10.1002/dmrr.2699

8. Lipsky, B. A., Berendt, A. R., Cornia, P. B., Pile, J. C., Peters, E. J., Armstrong, D. G., ... & Senneville, E. (2012). 2012 Infectious Diseases Society of America Clinical Practice Guideline for the Diagnosis and Treatment of Diabetic Foot Infections. *Clinical Infectious Diseases*, 54(12), e132–e173. https://doi.org/10.1093/cid/cis346

9. Lipsky, B. A., Senneville, É., Abbas, Z. G., Aragón-Sánchez, J., Diggle, M., Embil, J. M., ... & Peters, E. J. (2020). Guidelines on the diagnosis and treatment of foot infection in persons with diabetes (IWGDF 2019 update). *Diabetes/Metabolism Research and Reviews*, 36(S1), e3280. https://doi.org/10.1002/dmrr.3280

10. Mendel, J. M. (2017). *Uncertain Rule-Based Fuzzy Systems: Introduction and New Directions* (2nd ed.). Springer. https://doi.org/10.1007/978-3-319-51370-6

11. Secretaría de Salud. (2010). *Norma Oficial Mexicana NOM-015-SSA2-2010, Para la prevención, tratamiento y control de la diabetes mellitus*. Diario Oficial de la Federación. México.

12. Secretaría de Salud. (2008). *Norma Oficial Mexicana NOM-005-SSA3-2010 (antes SSA-005-08), Que establece los requisitos mínimos de infraestructura y equipamiento de establecimientos para la atención médica de pacientes ambulatorios*. Diario Oficial de la Federación. México.

13. Senneville, É., Albalawi, Z., van Asten, S. A., Abbas, Z. G., Allison, G., Aragón-Sánchez, J., ... & Lipsky, B. A. (2024). IWGDF/IDSA Guidelines on the Diagnosis and Treatment of Diabetes-related Foot Infections (IWGDF 2023 update). *Clinical Infectious Diseases*, 79(2), e1–e45. https://doi.org/10.1093/cid/ciad527

14. Sanderson, M., Howard, M. A., & Webster, J. (2014). Wound dressings for treating foot ulcers in people with diabetes: an overview of systematic reviews. *Cochrane Database of Systematic Reviews*. https://doi.org/10.1002/14651858.CD011393

15. van Netten, J. J., Bus, S. A., Apelqvist, J., Lipsky, B. A., Hinchliffe, R. J., Game, F., ... & Schaper, N. C. (2020). Definitions and criteria for diabetic foot disease. *Diabetes/Metabolism Research and Reviews*, 36(S1), e3268. https://doi.org/10.1002/dmrr.3268

---

## Anexo A: Información del sistema

| Campo | Valor |
|:------|:------|
| Nombre del sistema | Sistema Integral para Evaluación y Seguimiento del Pie Diabético |
| Versión | 5.0 |
| Framework | Flask (Python) |
| Motor de IA — Lógica Difusa | scikit-fuzzy (IT2-FLS implementación propia) |
| Motor de IA — Deep Learning | TensorFlow/Keras (MobileNetV2) |
| Generación de reportes | ReportLab |
| Base de datos | SQLite (Flask-SQLAlchemy) |
| Autora | Guadalupe Vélez Pérez |
| Año | 2026 |

---

## Anexo B: Glosario de términos

| Término | Definición |
|:--------|:-----------|
| **IT2-FLS** | Interval Type-2 Fuzzy Logic System. Sistema de lógica difusa tipo-2 intervalar. |
| **FOU** | Footprint of Uncertainty. Huella de incertidumbre que define el área entre las funciones de membresía superior e inferior. |
| **MF** | Membership Function. Función de membresía que define el grado de pertenencia de un valor a un conjunto difuso. |
| **Centroide** | Método de defuzzificación que calcula el centro de gravedad del área bajo la curva de salida difusa. |
| **MRSA** | Methicillin-Resistant *Staphylococcus aureus*. Estafilococo resistente a meticilina. |
| **BLEE** | Beta-Lactamasas de Espectro Extendido. Enzimas bacterianas que confieren resistencia a cefalosporinas de tercera generación. |
| **VO** | Vía oral. |
| **IV** | Vía intravenosa. |
| **SRIS** | Síndrome de Respuesta Inflamatoria Sistémica. |
| **IWGDF** | International Working Group on the Diabetic Foot. |
| **IDSA** | Infectious Diseases Society of America. |

---

*Documento generado como parte de la documentación técnica del Sistema Integral de Pie Diabético.*  
*© 2026 Guadalupe Vélez Pérez. Todos los derechos reservados.*
