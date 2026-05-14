# -*- coding: utf-8 -*-
"""
Módulo de generación de PDFs para el Sistema Integral de Pie Diabético.
Genera reportes clínicos (médico) y resúmenes simplificados (paciente).
Usa ReportLab para crear PDFs profesionales en formato A4.
"""

import os
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, Circle, String
from reportlab.graphics import renderPDF


# ── Colores del sistema ─────────────────────────────────────
COLOR_GRAVE = colors.HexColor('#d32f2f')
COLOR_MODERADA = colors.HexColor('#f57c00')
COLOR_LEVE = colors.HexColor('#fbc02d')
COLOR_NORMAL = colors.HexColor('#388e3c')
COLOR_PRIMARY = colors.HexColor('#1565c0')
COLOR_HEADER_BG = colors.HexColor('#1a237e')
COLOR_LIGHT_BG = colors.HexColor('#f5f5f5')
COLOR_TEXT = colors.HexColor('#212121')
COLOR_MUTED = colors.HexColor('#757575')

GRADO_COLORES = {1: COLOR_NORMAL, 2: COLOR_LEVE, 3: COLOR_MODERADA, 4: COLOR_GRAVE}
GRADO_LABELS = {1: 'No infectado', 2: 'Infección leve', 3: 'Infección moderada', 4: 'Infección grave'}
GRADO_EMOJI_TXT = {1: '●  NORMAL', 2: '●  LEVE', 3: '●  MODERADO', 4: '●  GRAVE'}

PROFUNDIDAD_LABELS = {0: 'Sin herida', 1: 'Superficial', 2: 'Profunda'}
ISQUEMIA_LABELS = {0: 'Sin isquemia', 1: 'Leve', 2: 'Crítica'}


def _get_styles():
    """Retorna estilos personalizados para los PDFs."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        'PDTitle', parent=styles['Title'],
        fontSize=18, textColor=colors.white, alignment=TA_CENTER,
        spaceAfter=6, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'PDSubtitle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#bbdefb'), alignment=TA_CENTER,
        spaceAfter=4, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'PDSectionTitle', parent=styles['Heading2'],
        fontSize=13, textColor=COLOR_PRIMARY, spaceBefore=14, spaceAfter=6,
        fontName='Helvetica-Bold', borderPadding=(0, 0, 2, 0),
        borderWidth=0, leftIndent=0
    ))
    styles.add(ParagraphStyle(
        'PDNormal', parent=styles['Normal'],
        fontSize=10, textColor=COLOR_TEXT, leading=14,
        fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'PDSmall', parent=styles['Normal'],
        fontSize=8, textColor=COLOR_MUTED, leading=10,
        fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'PDGradeResult', parent=styles['Heading1'],
        fontSize=22, alignment=TA_CENTER, spaceAfter=4,
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'PDCenter', parent=styles['Normal'],
        fontSize=10, alignment=TA_CENTER, textColor=COLOR_TEXT,
        fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'PDRecommendation', parent=styles['Normal'],
        fontSize=10, textColor=COLOR_TEXT, leading=14,
        fontName='Helvetica', leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4, backColor=COLOR_LIGHT_BG,
        borderPadding=8
    ))
    styles.add(ParagraphStyle(
        'PDFooter', parent=styles['Normal'],
        fontSize=7, textColor=COLOR_MUTED, alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    ))
    return styles


def _add_header(elements, styles, titulo, subtitulo=None):
    """Agrega un encabezado profesional con fondo de color."""
    header_data = [[Paragraph(titulo, styles['PDTitle'])]]
    if subtitulo:
        header_data.append([Paragraph(subtitulo, styles['PDSubtitle'])])

    header_table = Table(header_data, colWidths=[17 * cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_HEADER_BG),
        ('TOPPADDING', (0, 0), (-1, 0), 16),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))


def _add_footer(canvas, doc):
    """Pie de página con número de página y nombre del sistema."""
    canvas.saveState()
    canvas.setFont('Helvetica-Oblique', 7)
    canvas.setFillColor(COLOR_MUTED)
    footer_text = f"Sistema Integral de Pie Diabético v5.0 — Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} — Página {doc.page}"
    canvas.drawCentredString(A4[0] / 2, 15 * mm, footer_text)
    disclaimer = "Este documento es un apoyo diagnóstico y NO sustituye el criterio clínico profesional."
    canvas.drawCentredString(A4[0] / 2, 10 * mm, disclaimer)
    canvas.restoreState()


def _build_severity_box(grado, etiqueta, crisp_value, styles):
    """Crea una tabla-caja visual con el resultado de gravedad."""
    color = GRADO_COLORES.get(grado, COLOR_NORMAL)
    grade_text = f'<font color="{color.hexval()}"><b>Grado {grado}/4 — {etiqueta}</b></font>'
    crisp_text = f'Valor Crisp: {crisp_value:.3f}'

    data = [
        [Paragraph(grade_text, ParagraphStyle('grade', parent=styles['PDGradeResult'],
                                               fontSize=20, textColor=color))],
        [Paragraph(crisp_text, styles['PDCenter'])]
    ]
    t = Table(data, colWidths=[16 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
        ('BOX', (0, 0), (-1, -1), 2, color),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def _build_traffic_light(grado):
    """Crea un semáforo visual tipo dibujo."""
    d = Drawing(200, 50)
    positions = [(20, 25), (70, 25), (120, 25), (170, 25)]
    grade_colors = [COLOR_NORMAL, COLOR_LEVE, COLOR_MODERADA, COLOR_GRAVE]
    labels = ['Normal', 'Leve', 'Moderado', 'Grave']

    for i, (x, y) in enumerate(positions):
        fill_color = grade_colors[i] if (i + 1) == grado else colors.HexColor('#e0e0e0')
        opacity = 1.0 if (i + 1) == grado else 0.3
        c = Circle(x, y, 12)
        c.fillColor = fill_color
        c.strokeColor = colors.HexColor('#bdbdbd')
        c.strokeWidth = 1
        c.fillOpacity = opacity
        d.add(c)
        label = String(x, y - 22, labels[i], fontSize=7, textAnchor='middle',
                       fillColor=COLOR_MUTED)
        d.add(label)
    return d


# ═══════════════════════════════════════════════════════════════
#  PDF PARA MÉDICO — Reporte Clínico Completo
# ═══════════════════════════════════════════════════════════════

def generar_pdf_reporte_medico(evaluacion, paciente, medico, upload_folder=None):
    """
    Genera un PDF profesional de reporte clínico para el médico.
    
    Args:
        evaluacion: Objeto Evaluacion con los resultados
        paciente: Objeto Paciente
        medico: Objeto Usuario (médico)
        upload_folder: Ruta a la carpeta de uploads para imágenes
    
    Returns:
        BytesIO buffer con el PDF generado
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=2.5 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm
    )
    styles = _get_styles()
    elements = []

    # ── Encabezado ──
    _add_header(elements, styles,
                '🦶 REPORTE CLÍNICO — PIE DIABÉTICO',
                'Sistema Integral de Evaluación · IWGDF 2023 / NOM-015-SSA2-2010')

    # ── Datos del paciente y médico ──
    elements.append(Paragraph('<b>📋 DATOS DEL PACIENTE</b>', styles['PDSectionTitle']))
    patient_data = [
        ['Nombre:', paciente.nombre or '-',
         'Edad:', f"{paciente.edad or '-'} años"],
        ['Sexo:', paciente.sexo or '-',
         'Tipo Diabetes:', paciente.tipo_diabetes or '-'],
        ['Años diagnóstico:', str(paciente.anios_diagnostico or '-'),
         'Fecha evaluación:', evaluacion.fecha.strftime('%d/%m/%Y %H:%M') if evaluacion.fecha else '-'],
    ]
    pt = Table(patient_data, colWidths=[3.5 * cm, 5 * cm, 3.5 * cm, 5 * cm])
    pt.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), COLOR_MUTED),
        ('TEXTCOLOR', (2, 0), (2, -1), COLOR_MUTED),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#eeeeee')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 4))

    medico_data = [
        ['Médico evaluador:', medico.nombre_completo or '-',
         'Cédula:', medico.cedula_profesional or '-']
    ]
    mt = Table(medico_data, colWidths=[3.5 * cm, 5 * cm, 3.5 * cm, 5 * cm])
    mt.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), COLOR_MUTED),
        ('TEXTCOLOR', (2, 0), (2, -1), COLOR_MUTED),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(mt)
    elements.append(Spacer(1, 8))

    # ── Resultado principal ──
    elements.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=6))
    elements.append(Paragraph('<b>🔬 RESULTADO DE EVALUACIÓN</b>', styles['PDSectionTitle']))

    grado = evaluacion.grado or 1
    etiqueta = evaluacion.etiqueta or GRADO_LABELS.get(grado, 'N/D')
    crisp = evaluacion.gravedad_crisp or 0.0

    elements.append(_build_severity_box(grado, etiqueta, crisp, styles))
    elements.append(Spacer(1, 6))

    # Semáforo visual
    elements.append(_build_traffic_light(grado))
    elements.append(Spacer(1, 6))

    # Detalles difusos
    fuzzy_data = [
        ['Intervalo Tipo-2:', f'[{evaluacion.gravedad_lower:.3f}, {evaluacion.gravedad_upper:.3f}]',
         'Incertidumbre:', f'{evaluacion.incertidumbre:.4f}' if evaluacion.incertidumbre else '-'],
        ['Valor Crisp:', f'{crisp:.3f}',
         'Confianza:', evaluacion.confianza or '-']
    ]
    ft = Table(fuzzy_data, colWidths=[3.5 * cm, 5 * cm, 3.5 * cm, 5 * cm])
    ft.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), COLOR_MUTED),
        ('TEXTCOLOR', (2, 0), (2, -1), COLOR_MUTED),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
    ]))
    elements.append(ft)
    elements.append(Spacer(1, 8))

    # ── Parámetros clínicos evaluados ──
    elements.append(Paragraph('<b>📊 PARÁMETROS CLÍNICOS EVALUADOS</b>', styles['PDSectionTitle']))
    params_data = [
        ['Parámetro', 'Valor', 'Parámetro', 'Valor'],
        ['Signos locales:', f'{evaluacion.signos_locales}/4',
         'Eritema:', f'{evaluacion.eritema_cm} cm'],
        ['Profundidad:', PROFUNDIDAD_LABELS.get(evaluacion.profundidad, '-'),
         'Signos sistémicos:', 'Sí' if evaluacion.signos_sist else 'No'],
        ['Isquemia:', ISQUEMIA_LABELS.get(evaluacion.isquemia, '-'),
         'Glucosa:', f'{evaluacion.glucosa_mgdl} mg/dL'],
    ]
    cp = Table(params_data, colWidths=[3.5 * cm, 5 * cm, 3.5 * cm, 5 * cm])
    cp.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('TEXTCOLOR', (0, 1), (0, -1), COLOR_MUTED),
        ('TEXTCOLOR', (2, 1), (2, -1), COLOR_MUTED),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#eeeeee')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
    ]))
    elements.append(cp)
    elements.append(Spacer(1, 8))

    # ── Recomendación clínica ──
    elements.append(Paragraph('<b>💊 RECOMENDACIÓN CLÍNICA</b>', styles['PDSectionTitle']))
    recomendacion_text = (evaluacion.recomendacion or 'Sin recomendación').replace('\n', '<br/>')
    elements.append(Paragraph(recomendacion_text, styles['PDRecommendation']))
    elements.append(Spacer(1, 8))

    # ── Antibióticos sugeridos ──
    antibioticos = {
        1: "No requiere antibióticos.",
        2: "Cefalexina 500 mg c/6h VO × 7-14 días\nAlternativa: Amoxicilina-clavulánico 875/125 mg c/12h VO",
        3: "Amoxicilina-clavulánico 1 g c/8h IV o VO\nAlternativa: Clindamicina 600 mg c/8h + Ciprofloxacino 400 mg c/12h IV",
        4: "Piperacilina-tazobactam 4.5 g c/6h IV\nAlternativa: Meropenem 1 g c/8h IV + Vancomicina 15-20 mg/kg c/12h IV"
    }
    atb_text = antibioticos.get(grado, 'N/D')
    elements.append(Paragraph('<b>💉 TRATAMIENTO ANTIBIÓTICO SUGERIDO (IWGDF 2023 / NOM-015)</b>', styles['PDSectionTitle']))
    color_atb = GRADO_COLORES.get(grado, COLOR_NORMAL)
    atb_box_data = [[Paragraph(atb_text.replace('\n', '<br/>'), styles['PDNormal'])]]
    atb_table = Table(atb_box_data, colWidths=[16 * cm])
    atb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff8e1')),
        ('BOX', (0, 0), (-1, -1), 1.5, color_atb),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(atb_table)
    elements.append(Paragraph(
        '<i>⚠ Ajustar según cultivos y sensibilidad antimicrobiana.</i>',
        styles['PDSmall']))
    elements.append(Spacer(1, 8))

    # ── Resultado Deep Learning ──
    if evaluacion.dl_categoria:
        elements.append(Paragraph('<b>🧠 ANÁLISIS DEEP LEARNING (CNN - MobileNetV2)</b>', styles['PDSectionTitle']))
        dl_data = [
            ['Clasificación:', evaluacion.dl_categoria or '-'],
            ['Probabilidad Úlcera Diabética:', f'{(evaluacion.dl_probabilidad or 0) * 100:.1f}%'],
        ]
        dl_table = Table(dl_data, colWidths=[6 * cm, 11 * cm])
        dl_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), COLOR_MUTED),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT_BG),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ]))
        elements.append(dl_table)
        elements.append(Spacer(1, 8))

    # ── Imagen de la lesión ──
    if evaluacion.imagen_path and upload_folder:
        img_path = os.path.join(upload_folder, evaluacion.imagen_path)
        if os.path.exists(img_path):
            try:
                elements.append(Paragraph('<b>📷 IMAGEN DE LA LESIÓN</b>', styles['PDSectionTitle']))
                img = Image(img_path, width=8 * cm, height=8 * cm, kind='proportional')
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass

    # ── Notas clínicas ──
    if evaluacion.notas_clinicas:
        elements.append(Paragraph('<b>📝 NOTAS CLÍNICAS</b>', styles['PDSectionTitle']))
        elements.append(Paragraph(evaluacion.notas_clinicas, styles['PDNormal']))
        elements.append(Spacer(1, 8))

    # ── Disclaimer final ──
    elements.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_MUTED, spaceBefore=10, spaceAfter=6))
    elements.append(Paragraph(
        '<b>Aviso:</b> Este reporte es generado por un sistema de apoyo diagnóstico basado en IWGDF 2023, '
        'NOM-015-SSA2-2010 y guías IDSA. <b>No sustituye el criterio clínico profesional.</b> '
        'Las recomendaciones de antibióticos deben ajustarse según el cuadro clínico individual y cultivos.',
        styles['PDSmall']
    ))

    doc.build(elements, onFirstPage=_add_footer, onLaterPages=_add_footer)
    buffer.seek(0)
    return buffer


# ═══════════════════════════════════════════════════════════════
#  PDF PARA PACIENTE — Resumen Simplificado
# ═══════════════════════════════════════════════════════════════

def generar_pdf_resumen_paciente(paciente, seguimientos, ultimo_seguimiento=None):
    """
    Genera un PDF simplificado para el paciente con lenguaje sencillo.
    
    Args:
        paciente: Objeto Paciente
        seguimientos: Lista de últimos seguimientos
        ultimo_seguimiento: El seguimiento más reciente
    
    Returns:
        BytesIO buffer con el PDF generado
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=2.5 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm
    )
    styles = _get_styles()
    elements = []

    # ── Encabezado ──
    _add_header(elements, styles,
                '🦶 MI RESUMEN DE SALUD DEL PIE',
                'Información para el paciente · Pie Diabético')

    # ── Datos del paciente ──
    elements.append(Paragraph('<b>👤 MIS DATOS</b>', styles['PDSectionTitle']))
    p_data = [
        ['Nombre:', paciente.nombre or '-'],
        ['Edad:', f"{paciente.edad or '-'} años"],
        ['Tipo de diabetes:', paciente.tipo_diabetes or '-'],
        ['Fecha del resumen:', datetime.now().strftime('%d/%m/%Y %H:%M')],
    ]
    pt = Table(p_data, colWidths=[4.5 * cm, 12.5 * cm])
    pt.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), COLOR_MUTED),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 10))

    # ── Estado actual (semáforo) ──
    elements.append(Paragraph('<b>🚦 MI ESTADO ACTUAL</b>', styles['PDSectionTitle']))

    if ultimo_seguimiento:
        grado = ultimo_seguimiento.grado or 1
        etiqueta = ultimo_seguimiento.etiqueta or GRADO_LABELS.get(grado, '-')
        color = GRADO_COLORES.get(grado, COLOR_NORMAL)
        fecha_ult = ultimo_seguimiento.fecha.strftime('%d/%m/%Y %H:%M') if ultimo_seguimiento.fecha else '-'

        # Caja de estado
        estado_texts = {
            1: '🟢 Tu pie está BIEN — Sigue cuidándote',
            2: '🟡 Tu pie necesita ATENCIÓN — Consulta a tu médico en 24-48 horas',
            3: '🟠 URGENTE — Acude al médico HOY MISMO',
            4: '🔴 ¡EMERGENCIA! — Ve a URGENCIAS AHORA',
        }
        estado_text = estado_texts.get(grado, 'Sin información')

        status_data = [
            [Paragraph(f'<font size="14"><b>{estado_text}</b></font>',
                        ParagraphStyle('st', parent=styles['PDCenter'], textColor=color, fontSize=14))],
            [Paragraph(f'Último registro: {fecha_ult}', styles['PDCenter'])]
        ]
        st = Table(status_data, colWidths=[16 * cm])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
            ('BOX', (0, 0), (-1, -1), 3, color),
            ('TOPPADDING', (0, 0), (-1, 0), 16),
            ('BOTTOMPADDING', (0, -1), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(st)
        elements.append(Spacer(1, 6))

        # Semáforo visual
        elements.append(_build_traffic_light(grado))
        elements.append(Spacer(1, 10))

        # Datos registrados
        elements.append(Paragraph(f'<b>Datos de tu último registro ({fecha_ult}):</b>', styles['PDNormal']))
        seg_data = [
            ['Glucosa:', f'{ultimo_seguimiento.glucosa_mgdl or "-"} mg/dL'],
        ]
        if ultimo_seguimiento.temperatura:
            seg_data.append(['Temperatura:', f'{ultimo_seguimiento.temperatura}°C'])
        if ultimo_seguimiento.dolor_nivel is not None:
            seg_data.append(['Nivel de dolor:', f'{ultimo_seguimiento.dolor_nivel}/10'])
        sd = Table(seg_data, colWidths=[4.5 * cm, 12.5 * cm])
        sd.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), COLOR_MUTED),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(sd)
    else:
        elements.append(Paragraph('Aún no tienes registros de seguimiento.', styles['PDCenter']))

    elements.append(Spacer(1, 10))

    # ── Recomendaciones de cuidado ──
    elements.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=6))
    elements.append(Paragraph('<b>✅ RECOMENDACIONES DE CUIDADO DIARIO</b>', styles['PDSectionTitle']))

    cuidados = [
        '🧼 Lava tus pies diariamente con agua tibia y jabón suave',
        '🧴 Seca bien entre los dedos y aplica crema hidratante (NO entre los dedos)',
        '👀 Revisa tus pies cada día buscando heridas, ampollas o cambios de color',
        '👟 Usa zapatos cómodos y cerrados, calcetines de algodón sin costuras',
        '🚫 Nunca camines descalzo',
        '✂️ Corta las uñas rectas, sin redondear las esquinas',
        '🩹 Si tienes herida: limpia con solución salina, NO uses alcohol ni agua oxigenada',
        '📏 Controla tu glucosa según indicaciones médicas',
    ]
    for c in cuidados:
        elements.append(Paragraph(f'• {c}', styles['PDNormal']))
    elements.append(Spacer(1, 10))

    # ── Cuándo acudir al médico ──
    elements.append(Paragraph('<b>🚨 VE INMEDIATAMENTE A URGENCIAS SI:</b>', styles['PDSectionTitle']))
    alarmas = [
        'Tienes fiebre (más de 38°C) con herida en el pie',
        'Ves enrojecimiento que se extiende rápidamente',
        'Hay mal olor o pus en la herida',
        'Sientes dolor intenso que no mejora',
        'Tu pie está frío, morado o sin pulso',
        'Ves hueso o tejido profundo expuesto',
    ]
    alarm_items = []
    for a in alarmas:
        alarm_items.append([Paragraph(f'⚠️ {a}', ParagraphStyle('al', parent=styles['PDNormal'],
                                                                    textColor=COLOR_GRAVE))])
    alarm_table = Table(alarm_items, colWidths=[16 * cm])
    alarm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffebee')),
        ('BOX', (0, 0), (-1, -1), 2, COLOR_GRAVE),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(alarm_table)
    elements.append(Spacer(1, 10))

    # ── Historial reciente de seguimientos ──
    if seguimientos and len(seguimientos) > 0:
        elements.append(Paragraph('<b>📋 MIS ÚLTIMOS SEGUIMIENTOS</b>', styles['PDSectionTitle']))
        hist_header = [['Fecha', 'Estado', 'Glucosa']]
        hist_rows = []
        for s in seguimientos[:10]:  # últimos 10
            fecha_str = s.fecha.strftime('%d/%m/%Y') if s.fecha else '-'
            estado = s.etiqueta or GRADO_LABELS.get(s.grado, '-')
            glucosa = f'{s.glucosa_mgdl} mg/dL' if s.glucosa_mgdl else '-'
            hist_rows.append([fecha_str, estado, glucosa])

        hist_data = hist_header + hist_rows
        ht = Table(hist_data, colWidths=[4.5 * cm, 7 * cm, 5.5 * cm])
        ht.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#eeeeee')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(ht)
        elements.append(Spacer(1, 8))

    # ── Frecuencia de consultas recomendada ──
    elements.append(Paragraph('<b>📅 ¿CADA CUÁNDO DEBO IR AL MÉDICO?</b>', styles['PDSectionTitle']))
    freq_data = [
        ['Estado', 'Frecuencia de visita'],
        ['🟢 Sin heridas', 'Cada 3-6 meses'],
        ['🟡 Herida leve', 'Cada 1-2 semanas'],
        ['🟠 Herida moderada', 'Cada 2-3 días'],
        ['🔴 Herida grave', 'URGENCIAS INMEDIATA'],
    ]
    fq = Table(freq_data, colWidths=[6 * cm, 11 * cm])
    fq.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#eeeeee')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
    ]))
    elements.append(fq)
    elements.append(Spacer(1, 10))

    # ── Disclaimer ──
    elements.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_MUTED, spaceBefore=10, spaceAfter=6))
    elements.append(Paragraph(
        '<b>Importante:</b> Esta guía es informativa y NO reemplaza la consulta médica. '
        'Si tienes dudas sobre tu estado de salud, consulta siempre a tu médico. '
        'Basada en NOM-015-SSA2-2010 e IWGDF 2023.',
        styles['PDSmall']
    ))

    doc.build(elements, onFirstPage=_add_footer, onLaterPages=_add_footer)
    buffer.seek(0)
    return buffer
