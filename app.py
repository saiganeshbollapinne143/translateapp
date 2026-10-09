
import io
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from xml.sax.saxutils import escape

st.set_page_config(page_title="TXT to Translated PDF", page_icon="🌍")
st.title("🌍 TXT to Translated PDF")
st.write("Upload a text file, translate it, and download the translation as a PDF.")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHUNK_SIZE = 1000

LANGUAGES = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Arabic": "arb_Arab",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Russian": "rus_Cyrl",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym"
}

torch.set_num_threads(2)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def split_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > size and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

def make_pdf(text, language):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch
    )
    styles = getSampleStyleSheet()
    story = [Paragraph("Translated Document - " + escape(language), styles["Title"]),
             Spacer(1, 16)]
    for paragraph in text.splitlines():
        if paragraph.strip():
            story.append(Paragraph(escape(paragraph), styles["BodyText"]))
        else:
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 5))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

uploaded_file = st.file_uploader("Upload input TXT file", type=["txt"])

source_language = st.selectbox("Input language", list(LANGUAGES.keys()), index=0)
target_language = st.selectbox("Output language", list(LANGUAGES.keys()), index=1)

if uploaded_file:
    try:
        input_text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        input_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

    st.text_area("Input text preview", input_text, height=180)

    if st.button("Translate and Create PDF", type="primary"):
        if not input_text.strip():
            st.error("The uploaded TXT file is empty.")
            st.stop()
        if source_language == target_language:
            st.warning("Input and output languages are the same. The text will be exported without translation.")
            translated_text = input_text
        else:
            try:
                with st.spinner("Loading translation model..."):
                    tokenizer, model = load_model()

                chunks = split_text(input_text)
                progress = st.progress(0)
                status = st.empty()
                output_chunks = []

                tokenizer.src_lang = LANGUAGES[source_language]
                target_id = tokenizer.convert_tokens_to_ids(
                    LANGUAGES[target_language]
                )

                for i, chunk in enumerate(chunks):
                    status.write(f"Translating chunk {i + 1} of {len(chunks)}...")
                    inputs = tokenizer(
                        chunk, return_tensors="pt",
                        truncation=True, max_length=512
                    )
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            forced_bos_token_id=target_id,
                            max_new_tokens=384,
                            num_beams=2,
                            do_sample=False
                        )
                    output_chunks.append(
                        tokenizer.batch_decode(
                            generated, skip_special_tokens=True
                        )[0]
                    )
                    progress.progress((i + 1) / len(chunks))

                translated_text = "\n\n".join(output_chunks)
                status.success("Translation completed!")

            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.stop()

        pdf_bytes = make_pdf(translated_text, target_language)
        st.subheader("Translated text preview")
        st.text_area("Translation", translated_text, height=250)
        st.download_button(
            "⬇️ Download translated PDF",
            data=pdf_bytes,
            file_name=f"translated_{target_language.lower()}.pdf",
            mime="application/pdf",
            type="primary"
        )

import io
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from xml.sax.saxutils import escape

st.set_page_config(page_title="TXT to Translated PDF", page_icon="🌍")
st.title("🌍 TXT to Translated PDF")
st.write("Upload a text file, translate it, and download the translation as a PDF.")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHUNK_SIZE = 1000

LANGUAGES = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Arabic": "arb_Arab",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Russian": "rus_Cyrl",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym"
}

torch.set_num_threads(2)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def split_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > size and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

def make_pdf(text, language):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch
    )
    styles = getSampleStyleSheet()
    story = [Paragraph("Translated Document - " + escape(language), styles["Title"]),
             Spacer(1, 16)]
    for paragraph in text.splitlines():
        if paragraph.strip():
            story.append(Paragraph(escape(paragraph), styles["BodyText"]))
        else:
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 5))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

uploaded_file = st.file_uploader("Upload input TXT file", type=["txt"])

source_language = st.selectbox("Input language", list(LANGUAGES.keys()), index=0)
target_language = st.selectbox("Output language", list(LANGUAGES.keys()), index=1)

if uploaded_file:
    try:
        input_text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        input_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

    st.text_area("Input text preview", input_text, height=180)

    if st.button("Translate and Create PDF", type="primary"):
        if not input_text.strip():
            st.error("The uploaded TXT file is empty.")
            st.stop()
        if source_language == target_language:
            st.warning("Input and output languages are the same. The text will be exported without translation.")
            translated_text = input_text
        else:
            try:
                with st.spinner("Loading translation model..."):
                    tokenizer, model = load_model()

                chunks = split_text(input_text)
                progress = st.progress(0)
                status = st.empty()
                output_chunks = []

                tokenizer.src_lang = LANGUAGES[source_language]
                target_id = tokenizer.convert_tokens_to_ids(
                    LANGUAGES[target_language]
                )

                for i, chunk in enumerate(chunks):
                    status.write(f"Translating chunk {i + 1} of {len(chunks)}...")
                    inputs = tokenizer(
                        chunk, return_tensors="pt",
                        truncation=True, max_length=512
                    )
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            forced_bos_token_id=target_id,
                            max_new_tokens=384,
                            num_beams=2,
                            do_sample=False
                        )
                    output_chunks.append(
                        tokenizer.batch_decode(
                            generated, skip_special_tokens=True
                        )[0]
                    )
                    progress.progress((i + 1) / len(chunks))

                translated_text = "\n\n".join(output_chunks)
                status.success("Translation completed!")

            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.stop()

        pdf_bytes = make_pdf(translated_text, target_language)
        st.subheader("Translated text preview")
        st.text_area("Translation", translated_text, height=250)
        st.download_button(
            "⬇️ Download translated PDF",
            data=pdf_bytes,
            file_name=f"translated_{target_language.lower()}.pdf",
            mime="application/pdf",
            type="primary"
        )
