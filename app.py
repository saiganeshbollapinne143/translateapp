import streamlit as st
import torch
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Translator", page_icon="🌐")

MODEL = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn"
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- EXTRACT TEXT ----------------
def extract_text(file):
    name = file.name.lower()

    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

# ---------------- SPLIT TEXT ----------------
def split_text(text, size=800):
    words = text.split()
    chunks, current = [], ""

    for word in words:
        if len(current) + len(word) + 1 <= size:
            current += (" " if current else "") + word
        else:
            if current:
                chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks

# ---------------- TRANSLATE ----------------
def translate(text, source_code, target_code):

    chunks = split_text(text)

    results = []

    progress = st.progress(0)
    status = st.empty()

    tokenizer.src_lang = source_code
    target_id = tokenizer.convert_tokens_to_ids(target_code)

    for i, chunk in enumerate(chunks):

        status.write(f"Translating {i + 1} / {len(chunks)}...")

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False
            )

        result = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

        results.append(result)

        progress.progress((i + 1) / len(chunks))

    status.success("Translation completed!")

    return "\n\n".join(results)

# ---------------- PDF ----------------
def make_pdf(text):

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    margin = 40
    y = height - margin

    pdf.setFont("Helvetica", 10)

    for paragraph in text.split("\n"):

        words = paragraph.split()
        line = ""

        for word in words:

            test = line + " " + word if line else word

            if pdf.stringWidth(test, "Helvetica", 10) < width - 80:
                line = test

            else:
                pdf.drawString(margin, y, line)
                y -= 15
                line = word

                if y < margin:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = height - margin

        if line:
            pdf.drawString(margin, y, line)
            y -= 15

        y -= 5

        if y < margin:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - margin

    pdf.save()
    buffer.seek(0)

    return buffer

# ---------------- UI ----------------
st.title("🌐 AI Document Translator")

st.write("Translate text or documents using NLLB-200.")

col1, col2 = st.columns(2)

with col1:
    source = st.selectbox(
        "Source Language",
        list(LANGUAGES.keys())
    )

with col2:
    target = st.selectbox(
        "Target Language",
        list(LANGUAGES.keys()),
        index=1
    )

text = st.text_area(
    "Enter text",
    height=180
)

file = st.file_uploader(
    "Upload TXT, PDF or DOCX",
    type=["txt", "pdf", "docx"]
)

if st.button("🔄 Translate", type="primary"):

    if file:
        text = extract_text(file)

    if not text.strip():
        st.warning("Please enter text or upload a document.")
        st.stop()

    translated = translate(
        text,
        LANGUAGES[source],
        LANGUAGES[target]
    )

    st.subheader("Translated Text")

    st.text_area(
        "Result",
        translated,
        height=300
    )

    st.download_button(
        "📄 Download TXT",
        translated,
        file_name=f"translated_{target}.txt",
        mime="text/plain"
    )

    pdf = make_pdf(translated)

    st.download_button(
        "📕 Download Translated PDF",
        pdf,
        file_name=f"translated_{target}.pdf",
        mime="application/pdf"
    )