import streamlit as st
from datetime import datetime
import io
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# Put this section after translation completes and
# st.session_state.translated contains the Tamil output.

if st.session_state.translated:
    result = st.session_state.translated
    st.subheader("📄 Download Tamil PDF")

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CompanyTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#2458B8"),
        fontSize=18,
        leading=24,
        spaceAfter=10
    ))

    story = [
        Paragraph("AIT GLOBAL TECHNOLOGIES", styles["CompanyTitle"]),
        Paragraph("English to Tamil Translation", styles["Heading2"]),
        Paragraph(
            "Generated: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            styles["Normal"]
        ),
        Spacer(1, 20)
    ]

    for paragraph in result.splitlines():
        if paragraph.strip():
            story.append(Paragraph(escape(paragraph), styles["Normal"]))
            story.append(Spacer(1, 8))

    try:
        doc.build(story)
        pdf_bytes = pdf_buffer.getvalue()
        st.download_button(
            "⬇️ Download translated PDF",
            data=pdf_bytes,
            file_name="AIT_English_to_Tamil.pdf",
            mime="application/pdf",
            key="download_tamil_pdf"
        )
    except Exception as e:
        st.error(f"PDF creation failed: {e}")

