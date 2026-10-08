import streamlit as st
import torch
import chromadb
import uuid
from io import BytesIO
from datetime import datetime
from pypdf import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Translator",
    page_icon="🌐",
    layout="wide"
)

MODEL = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "/tmp/chroma_db"

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

# =========================================================
# CHROMADB
# =========================================================

@st.cache_resource
def load_database():

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name="translation_records"
    )

    return collection


collection = load_database()

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# =========================================================
# THREAD ID
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌐 AI Translator")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Translator", "📚 History"]
)

st.sidebar.divider()

st.sidebar.write("**Current Thread**")
st.sidebar.code(
    st.session_state.thread_id
)

if st.sidebar.button("🆕 New Conversation"):

    st.session_state.thread_id = str(
        uuid.uuid4()
    )

    st.rerun()

# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text(file):

    name = file.name.lower()

    if name.endswith(".txt"):

        return file.read().decode(
            "utf-8",
            errors="ignore"
        )

    if name.endswith(".pdf"):

        reader = PdfReader(file)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if name.endswith(".docx"):

        doc = Document(file)

        return "\n".join(
            p.text
            for p in doc.paragraphs
        )

    return ""


# =========================================================
# SPLIT TEXT
# =========================================================

def split_text(text, size=1200):

    words = text.split()

    chunks = []
    current = ""

    for word in words:

        if len(current) + len(word) + 1 <= size:

            current += (
                " " if current else ""
            ) + word

        else:

            if current:
                chunks.append(current)

            current = word

    if current:
        chunks.append(current)

    return chunks


# =========================================================
# TRANSLATION
# =========================================================

def translate(
    text,
    source_code,
    target_code
):

    chunks = split_text(text)

    results = []

    progress = st.progress(0)

    status = st.empty()

    tokenizer.src_lang = source_code

    target_id = tokenizer.convert_tokens_to_ids(
        target_code
    )

    for i, chunk in enumerate(chunks):

        status.write(
            f"Translating {i + 1} / {len(chunks)}..."
        )

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():

            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False
            )

        translated = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

        results.append(translated)

        progress.progress(
            (i + 1) / len(chunks)
        )

    progress.empty()

    status.success(
        "Translation completed!"
    )

    return "\n\n".join(results)


# =========================================================
# SAVE RECORD
# =========================================================

def save_record(
    source_text,
    translated_text,
    source_language,
    target_language
):

    collection.add(

        ids=[str(uuid.uuid4())],

        documents=[translated_text],

        metadatas=[{

            "source_text": source_text[:4000],

            "source_language":
                source_language,

            "target_language":
                target_language,

            "thread_id":
                st.session_state.thread_id,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }]
    )


# =========================================================
# PDF
# =========================================================

def create_pdf(text):

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    margin = 40

    y = height - margin

    pdf.setFont(
        "Helvetica",
        10
    )

    for paragraph in text.split("\n"):

        line = ""

        for word in paragraph.split():

            test = (
                line + " " + word
                if line
                else word
            )

            if pdf.stringWidth(
                test,
                "Helvetica",
                10
            ) <= width - 80:

                line = test

            else:

                pdf.drawString(
                    margin,
                    y,
                    line
                )

                y -= 15

                line = word

                if y < margin:

                    pdf.showPage()

                    pdf.setFont(
                        "Helvetica",
                        10
                    )

                    y = height - margin

        if line:

            pdf.drawString(
                margin,
                y,
                line
            )

            y -= 15

        if y < margin:

            pdf.showPage()

            pdf.setFont(
                "Helvetica",
                10
            )

            y = height - margin

    pdf.save()

    buffer.seek(0)

    return buffer


# =========================================================
# TRANSLATOR PAGE
# =========================================================

if page == "🏠 Translator":

    st.title("🌐 AI Translator")

    st.caption(
        "Translate text and documents using AI"
    )

    col1, col2 = st.columns(2)

    with col1:

        source_name = st.selectbox(
            "Source Language",
            list(LANGUAGES.keys())
        )

    with col2:

        target_name = st.selectbox(
            "Target Language",
            list(LANGUAGES.keys()),
            index=1
        )

    text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Type or paste your text here..."
    )

    file = st.file_uploader(
        "Upload TXT, PDF or DOCX",
        type=["txt", "pdf", "docx"]
    )

    if st.button(
        "🔄 Translate",
        type="primary",
        use_container_width=True
    ):

        if file:

            text = extract_text(file)

        if not text.strip():

            st.warning(
                "Enter text or upload a document."
            )

            st.stop()

        translated = translate(
            text,
            LANGUAGES[source_name],
            LANGUAGES[target_name]
        )

        save_record(
            text,
            translated,
            source_name,
            target_name
        )

        st.subheader(
            "✅ Translated Text"
        )

        st.text_area(
            "Output",
            translated,
            height=300
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "📄 Download TXT",
                translated,
                file_name=
                    f"translated_{target_name}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:

            pdf = create_pdf(
                translated
            )

            st.download_button(
                "📕 Download PDF",
                pdf,
                file_name=
                    f"translated_{target_name}.pdf",
                mime="application/pdf",
                use_container_width=True
            )


# =========================================================
# HISTORY PAGE
# =========================================================

if page == "📚 History":

    st.title("📚 Translation History")

    st.caption(
        "All saved translation records from ChromaDB"
    )

    records = collection.get()

    documents = records.get(
        "documents",
        []
    )

    metadatas = records.get(
        "metadatas",
        []
    )

    # ---------------------------------------------
    # HISTORY COUNT
    # ---------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            len(documents)
        )

    with col2:

        st.metric(
            "Current Thread",
            "Active"
        )

    with col3:

        st.metric(
            "Storage",
            "ChromaDB"
        )

    st.divider()

    # ---------------------------------------------
    # SEARCH
    # ---------------------------------------------

    search = st.text_input(
        "🔎 Search translation history",
        placeholder="Search saved translations..."
    )

    # ---------------------------------------------
    # FILTER
    # ---------------------------------------------

    if search:

        results = collection.query(
            query_texts=[search],
            n_results=20
        )

        documents = (
            results.get("documents", [[]])[0]
        )

        metadatas = (
            results.get("metadatas", [[]])[0]
        )

    # ---------------------------------------------
    # DISPLAY HISTORY
    # ---------------------------------------------

    if not documents:

        st.info(
            "No translation records found."
        )

    else:

        for i in range(len(documents)):

            metadata = metadatas[i]

            source = metadata.get(
                "source_language",
                "Unknown"
            )

            target = metadata.get(
                "target_language",
                "Unknown"
            )

            timestamp = metadata.get(
                "timestamp",
                ""
            )

            thread = metadata.get(
                "thread_id",
                ""
            )

            original = metadata.get(
                "source_text",
                ""
            )

            translated = documents[i]

            # =====================================
            # VISUAL CARD
            # =====================================

            with st.container(border=True):

                col1, col2 = st.columns(
                    [3, 1]
                )

                with col1:

                    st.markdown(
                        f"### 🌐 {source} → {target}"
                    )

                with col2:

                    st.caption(
                        timestamp
                    )

                st.divider()

                left, right = st.columns(2)

                with left:

                    st.markdown(
                        "**Original Text**"
                    )

                    st.write(
                        original
                    )

                with right:

                    st.markdown(
                        "**Translated Text**"
                    )

                    st.write(
                        translated
                    )

                st.caption(
                    f"🧵 Thread ID: {thread}"
                )

    st.divider()

    # ---------------------------------------------
    # CLEAR HISTORY
    # ---------------------------------------------

    if documents:

        if st.button(
            "🗑️ Clear All History",
            type="secondary"
        ):

            collection.delete(
                where={}
            )

            st.success(
                "Translation history cleared."
            )

            st.rerun()