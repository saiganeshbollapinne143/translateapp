import io
import time
import uuid
from functools import wraps
from datetime import datetime
from pathlib import Path

import streamlit as st
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

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


# ============================================================
# DECORATORS
# ============================================================

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
                f"❌ Error in {func.__name__}: {e}"
            )

            return None

    return wrapper


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_chroma():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="translation_history"
    )

    return collection


# ============================================================
# THREAD ID
# ============================================================

def get_thread_id():

    if "thread_id" not in st.session_state:

        st.session_state.thread_id = str(
            uuid.uuid4()
        )

    return st.session_state.thread_id


# ============================================================
# SAVE TRANSLATION
# ============================================================

@handle_errors
def save_translation(
    source_text,
    translated_text,
    source_language,
    target_language
):

    collection = get_chroma()

    thread_id = get_thread_id()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    translation_id = str(uuid.uuid4())

    collection.add(

        ids=[translation_id],

        documents=[translated_text],

        metadatas=[{

            "source_text": source_text,

            "source_language": source_language,

            "target_language": target_language,

            "thread_id": thread_id,

            "timestamp": timestamp

        }]
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading NLLB translation model..."
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

    model = model.to(device)

    return tokenizer, model, device


# ============================================================
# SPLIT TEXT
# ============================================================

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

        else:

            words = paragraph.split()

            current = ""

            for word in words:

                if (
                    len(current)
                    + len(word)
                    + 1
                    <= max_chars
                ):

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


# ============================================================
# TRANSLATION
# ============================================================

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

    tokenizer.src_lang = source_language

    target_id = tokenizer.convert_tokens_to_ids(
        target_language
    )

    results = []

    batch_size = (
        8
        if device == "cuda"
        else 4
    )

    for start in range(
        0,
        len(chunks),
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
        ).to(device)

        with torch.inference_mode():

            output = model.generate(

                **inputs,

                forced_bos_token_id=target_id,

                max_new_tokens=384,

                num_beams=1,

                do_sample=False
            )

        translated = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )

        results.extend(translated)

    return "\n\n".join(results)


# ============================================================
# READ FILE
# ============================================================

@handle_errors
def read_file(uploaded_file):

    extension = (
        uploaded_file.name
        .lower()
        .split(".")[-1]
    )

    data = uploaded_file.getvalue()

    if extension == "txt":

        return data.decode(
            "utf-8",
            errors="ignore"
        )

    if extension == "pdf":

        from pypdf import PdfReader

        reader = PdfReader(
            io.BytesIO(data)
        )

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

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
        "Only TXT, PDF and DOCX files are supported."
    )


# ============================================================
# HISTORY
# ============================================================

@handle_errors
def show_history(search_text=""):

    collection = get_chroma()

    total = collection.count()

    if total == 0:

        st.info(
            "📝 No previous translations yet."
        )

        return

    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    history = list(
        zip(
            documents,
            metadatas
        )
    )

    # Newest first
    history.reverse()

    if search_text:

        search_text = (
            search_text
            .lower()
            .strip()
        )

        history = [

            item

            for item in history

            if (
                search_text
                in str(item[0]).lower()
                or
                search_text
                in str(item[1]).lower()
            )
        ]

    if not history:

        st.warning(
            "No matching history found."
        )

        return

    for index, (
        translated_text,
        metadata
    ) in enumerate(history):

        source_language = metadata.get(
            "source_language",
            "Unknown"
        )

        target_language = metadata.get(
            "target_language",
            "Unknown"
        )

        timestamp = metadata.get(
            "timestamp",
            "Unknown"
        )

        thread_id = metadata.get(
            "thread_id",
            "Unknown"
        )

        source_text = metadata.get(
            "source_text",
            ""
        )

        with st.expander(

            f"🌐 {source_language} → "
            f"{target_language} | "
            f"{timestamp}"

        ):

            st.caption(
                f"🧵 Thread ID: {thread_id}"
            )

            st.markdown(
                "**Original Text**"
            )

            st.text_area(
                "Original",
                source_text,
                height=120,
                key=f"source_{index}"
            )

            st.markdown(
                "**Translation**"
            )

            st.text_area(
                "Translated",
                translated_text,
                height=180,
                key=f"translated_{index}"
            )

            st.download_button(

                "📥 Download Translation",

                translated_text.encode(
                    "utf-8"
                ),

                file_name=(
                    f"translation_{index}.txt"
                ),

                mime="text/plain",

                key=f"download_{index}",

                use_container_width=True
            )


# ============================================================
# CLEAR HISTORY
# ============================================================

@handle_errors
def clear_history():

    collection = get_chroma()

    data = collection.get()

    ids = data.get(
        "ids",
        []
    )

    if ids:

        collection.delete(
            ids=ids
        )

    st.success(
        "🗑️ Translation history cleared."
    )

    st.rerun()


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(

    page_title="AI Translator",

    page_icon="🌐",

    layout="wide"
)


# ============================================================
# SESSION
# ============================================================

thread_id = get_thread_id()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌐 AI Translator"
)

st.caption(
    "NLLB-200 Multilingual Translation "
    "with ChromaDB History"
)


# ============================================================
# LANGUAGE SELECTION
# ============================================================

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


# ============================================================
# INPUT
# ============================================================

text = st.text_area(

    "Enter text",

    height=180,

    placeholder=(
        "Type or paste your text here..."
    )
)


uploaded_file = st.file_uploader(

    "📁 Upload TXT / PDF / DOCX",

    type=[
        "txt",
        "pdf",
        "docx"
    ]
)


# ============================================================
# TRANSLATE BUTTON
# ============================================================

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

            target_name
        )

        st.subheader(
            "✅ Translation"
        )

        st.text_area(

            "Output",

            translated,

            height=300
        )

        st.download_button(

            "📥 Download TXT",

            translated.encode(
                "utf-8"
            ),

            file_name="translation.txt",

            mime="text/plain",

            use_container_width=True
        )

        st.success(
            "Translation completed and saved to history."
        )


# ============================================================
# CURRENT THREAD
# ============================================================

with st.expander(
    "🧵 Current Thread Information"
):

    st.write(
        f"**Thread ID:** `{thread_id}`"
    )

    st.write(
        f"**ChromaDB Records:** "
        f"`{get_chroma().count()}`"
    )


# ============================================================
# PREVIOUS HISTORY
# ============================================================

st.divider()

st.subheader(
    "🕘 Previous Translation History"
)


search_history = st.text_input(

    "🔍 Search History",

    placeholder=(
        "Search language, text, "
        "thread ID, etc."
    )
)


col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🔎 Search",
        use_container_width=True
    ):

        show_history(
            search_history
        )

with col2:

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        clear_history()


if not search_history:

    show_history()