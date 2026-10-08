import time
import io
import uuid
import chromadb
from functools import wraps
from pathlib import Path
from datetime import datetime

import streamlit as st


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).parent
MODEL_NAME = "facebook/nllb-200-distilled-600M"

CHROMA_DIR = BASE_DIR / "chroma_db"

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
    "Italian": "ita_Latn",
}


# =========================================================
# DECORATORS
# =========================================================

def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        st.caption(
            f"⏱️ {func.__name__}: {elapsed:.2f} seconds"
        )

        return result

    return wrapper


def handle_errors(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)

        except Exception as e:

            st.error(
                f"❌ {func.__name__}: {e}"
            )

            return None

    return wrapper


# =========================================================
# CHROMADB
# =========================================================

@st.cache_resource
def get_chroma():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="translations"
    )

    return collection


@handle_errors
def save_translation(
    source,
    translated,
    source_language,
    target_language,
    thread_id
):

    collection = get_chroma()

    translation_id = str(uuid.uuid4())

    collection.add(
        ids=[translation_id],

        documents=[translated],

        metadatas=[{
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "source_language": source_language,
            "target_language": target_language,
            "thread_id": thread_id,
            "source_text": source
        }]
    )


# =========================================================
# MODEL
# =========================================================

@st.cache_resource(
    show_spinner="Loading NLLB-200 model..."
)
def load_model():

    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()
    model.to(device)

    return tokenizer, model, device


# =========================================================
# TEXT CHUNKING
# =========================================================

@timer
def split_text(text, max_chars=500):

    paragraphs = [
        p.strip()
        for p in text.splitlines()
        if p.strip()
    ]

    chunks = []

    for paragraph in paragraphs:

        if len(paragraph) <= max_chars:

            chunks.append(paragraph)

            continue

        words = paragraph.split()

        current = ""

        for word in words:

            if len(current) + len(word) + 1 <= max_chars:

                current += " " + word

            else:

                if current:

                    chunks.append(
                        current.strip()
                    )

                current = word

        if current:

            chunks.append(
                current.strip()
            )

    return chunks


# =========================================================
# TRANSLATION
# =========================================================

@timer
@handle_errors
def translate_text(
    text,
    source_language,
    target_language
):

    import torch

    tokenizer, model, device = load_model()

    chunks = split_text(text)

    if not chunks:
        return None

    tokenizer.src_lang = source_language

    target_id = tokenizer.convert_tokens_to_ids(
        target_language
    )

    results = []

    batch_size = (
        16
        if device == "cuda"
        else 4
    )

    progress = st.progress(0)

    total = len(chunks)

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=384,
                num_beams=1,
                do_sample=False
            )

        results.extend(
            tokenizer.batch_decode(
                output,
                skip_special_tokens=True
            )
        )

        progress.progress(
            min(
                (start + len(batch)) / total,
                1.0
            )
        )

    progress.empty()

    return "\n\n".join(results)


# =========================================================
# FILE READER
# =========================================================

@handle_errors
def read_file(uploaded_file):

    extension = (
        uploaded_file.name
        .lower()
        .split(".")[-1]
    )

    data = uploaded_file.getvalue()

    # TXT
    if extension == "txt":

        return data.decode(
            "utf-8",
            errors="ignore"
        )

    # PDF
    if extension == "pdf":

        from pypdf import PdfReader

        reader = PdfReader(
            io.BytesIO(data)
        )

        pages = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                pages.append(page_text)

        return "\n".join(pages)

    # DOCX
    if extension == "docx":

        from docx import Document

        document = Document(
            io.BytesIO(data)
        )

        return "\n".join(
            p.text
            for p in document.paragraphs
            if p.text.strip()
        )

    raise ValueError(
        "Only TXT, PDF and DOCX are supported."
    )


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Translator",
    page_icon="🌐",
    layout="wide"
)


# =========================================================
# SESSION
# =========================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


# =========================================================
# UI
# =========================================================

st.title("🌐 AI Translator")

st.caption(
    "Fast multilingual translation using NLLB-200"
)

st.divider()


# =========================================================
# LANGUAGE
# =========================================================

col1, col2 = st.columns(2)

source_name = col1.selectbox(
    "Source Language",
    list(LANGUAGES.keys())
)

target_name = col2.selectbox(
    "Target Language",
    list(LANGUAGES.keys()),
    index=3
)


# =========================================================
# INPUT
# =========================================================

text = st.text_area(
    "Enter text",
    height=180,
    placeholder="Enter text to translate..."
)

uploaded_file = st.file_uploader(
    "Upload TXT / PDF / DOCX",
    type=["txt", "pdf", "docx"]
)


# =========================================================
# TRANSLATE
# =========================================================

if st.button(
    "🚀 Translate",
    type="primary",
    use_container_width=True
):

    if uploaded_file:

        text = read_file(
            uploaded_file
        )

    if not text:

        st.warning(
            "Please enter text or upload a file."
        )

        st.stop()

    if source_name == target_name:

        st.warning(
            "Please select different languages."
        )

        st.stop()

    translated = translate_text(
        text,
        LANGUAGES[source_name],
        LANGUAGES[target_name]
    )

    if translated:

        save_translation(
            text,
            translated,
            source_name,
            target_name,
            st.session_state.thread_id
        )

        st.divider()

        st.subheader(
            "✅ Translation"
        )

        st.text_area(
            "Translated Output",
            translated,
            height=350
        )

        st.download_button(
            "📄 Download TXT",
            translated.encode("utf-8"),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.success(
            "Translation completed successfully!"
        )


# =========================================================
# CHROMADB INFO
# =========================================================

with st.expander("ChromaDB Information"):

    try:

        collection = get_chroma()

        st.write(
            f"Stored translations: **{collection.count()}**"
        )

        st.write(
            f"Thread ID: `{st.session_state.thread_id}`"
        )

    except Exception as e:

        st.error(str(e))