import csv
import io
import importlib.metadata as md
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import chromadb
import pandas as pd
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Translator",
    page_icon="🌐",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = str(BASE_DIR / "chroma_db")

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

LATIN_FONT = BASE_DIR / "NotoSans-Regular.ttf"

SCRIPT_FONTS = {
    "Tamil": "NotoSansTamil-Regular.ttf",
    "Hindi": "NotoSansDevanagari-Regular.ttf",
    "Telugu": "NotoSansTelugu-Regular.ttf",
    "Kannada": "NotoSansKannada-Regular.ttf",
    "Malayalam": "NotoSansMalayalam-Regular.ttf",
}

FONT_BASE_URLS = [
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts",
    "https://github.com/notofonts/notofonts.github.io/raw/main/fonts",
]


# =========================================================
# LOAD CHROMADB
# =========================================================

@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name="translation_records"
    )


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


collection = load_database()
tokenizer, model = load_model()


# =========================================================
# SESSION
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# =========================================================
# FILE EXTRACTION
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
            paragraph.text
            for paragraph in doc.paragraphs
        )

    return ""


# =========================================================
# SPLIT TEXT
# =========================================================

def split_text(text, size=700):

    chunks = []
    current = ""

    for paragraph in text.split("\n"):

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        words = paragraph.split()

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

def translate(text, source_code, target_code):

    chunks = split_text(text)

    if not chunks:
        return ""

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
            max_length=256,
        ).to(DEVICE)

        with torch.no_grad():

            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=2,
                do_sample=False,
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
    status.success("Translation completed!")

    return "\n\n".join(results)


# =========================================================
# SAVE HISTORY
# =========================================================

def save_record(
    source_text,
    translated_text,
    source_language,
    target_language,
):

    collection.add(

        ids=[str(uuid.uuid4())],

        documents=[translated_text],

        metadatas=[
            {
                "source_text": source_text[:4000],
                "source_language": source_language,
                "target_language": target_language,
                "thread_id": st.session_state.thread_id,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        ],
    )


# =========================================================
# CREATE PDF
# =========================================================

def create_pdf(text, language):

    script_font = SCRIPT_FONTS.get(language)

    script_path = (
        BASE_DIR / script_font
        if script_font
        else None
    )

    if script_path and not script_path.exists():
        return None

    pdf = FPDF()
    pdf.add_page()

    latin_name = None

    if LATIN_FONT.exists():

        pdf.add_font(
            "Latin",
            "",
            str(LATIN_FONT)
        )

        latin_name = "Latin"

    elif script_path:

        return None

    if script_path:

        pdf.add_font(
            "Script",
            "",
            str(script_path)
        )

        pdf.set_font(
            "Script",
            size=12
        )

        if latin_name:
            pdf.set_fallback_fonts(["Latin"])

    elif latin_name:

        pdf.set_font(
            "Latin",
            size=12
        )

    else:

        pdf.set_font(
            "Helvetica",
            size=12
        )

    try:

        pdf.set_text_shaping(True)

    except Exception:
        pass

    pdf.multi_cell(
        0,
        8,
        text,
        new_x="LMARGIN",
        new_y="NEXT",
    )

    return bytes(pdf.output())


# =========================================================
# CREATE DOCX
# =========================================================

def create_docx(text):

    doc = Document()

    for paragraph in text.split("\n"):

        doc.add_paragraph(
            paragraph
        )

    output = io.BytesIO()

    doc.save(output)

    return output.getvalue()


# =========================================================
# DOWNLOAD FONTS
# =========================================================

def download_fonts():

    targets = [
        LATIN_FONT.name
    ] + list(SCRIPT_FONTS.values())

    results = []

    for fname in targets:

        destination = BASE_DIR / fname

        if destination.exists():

            results.append(
                (
                    fname,
                    True,
                    "already available"
                )
            )

            continue

        family = fname.rsplit(
            "-",
            1
        )[0]

        success = False
        error = ""

        for base in FONT_BASE_URLS:

            url = (
                f"{base}/{family}/"
                f"hinted/ttf/{fname}"
            )

            try:

                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    },
                )

                with urllib.request.urlopen(
                    request,
                    timeout=60
                ) as response:

                    data = response.read()

                if (
                    len(data) > 10000
                    and data[:4]
                    in (
                        b"\x00\x01\x00\x00",
                        b"true",
                    )
                ):

                    destination.write_bytes(data)

                    success = True
                    break

                error = (
                    "Downloaded file is not a valid TTF"
                )

            except Exception as exc:

                error = str(exc)

        results.append(
            (
                fname,
                success,
                "downloaded"
                if success
                else f"failed: {error}",
            )
        )

    return results


# =========================================================
# LOAD HISTORY
# =========================================================

def load_records(search=""):

    total = collection.count()

    if total == 0:
        return []

    if search:

        result = collection.query(
            query_texts=[search],
            n_results=min(20, total),
        )

        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]

    else:

        result = collection.get()

        ids = result["ids"]
        docs = result["documents"]
        metas = result["metadatas"]

    records = [

        {
            "id": record_id,
            "translated": document,
            **(metadata or {}),
        }

        for record_id, document, metadata
        in zip(ids, docs, metas)
    ]

    if not search:

        records.sort(
            key=lambda x: x.get(
                "timestamp",
                ""
            ),
            reverse=True,
        )

    return records


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌐 AI Translator")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Translator",
        "📚 History",
        "📊 Usage",
        "ℹ️ Requirements & Guide",
    ],
)

st.sidebar.divider()

st.sidebar.write("**Current Thread**")

st.sidebar.code(
    st.session_state.thread_id
)

if st.sidebar.button(
    "🆕 New Conversation"
):

    st.session_state.thread_id = str(
        uuid.uuid4()
    )

    st.session_state.pop(
        "last_result",
        None
    )

    st.rerun()


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
            list(LANGUAGES.keys()),
        )

    with col2:

        target_name = st.selectbox(
            "Target Language",
            list(LANGUAGES.keys()),
            index=1,
        )

    text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Type or paste your text here...",
    )

    file = st.file_uploader(
        "Upload TXT, PDF or DOCX",
        type=[
            "txt",
            "pdf",
            "docx",
        ],
    )

    if st.button(
        "🔄 Translate",
        type="primary",
        use_container_width=True,
    ):

        if file:

            text = extract_text(file)

        if not text.strip():

            st.warning(
                "Enter text or upload a document."
            )

            st.stop()

        if source_name == target_name:

            st.warning(
                "Source and target languages are the same."
            )

            st.stop()

        translated = translate(
            text,
            LANGUAGES[source_name],
            LANGUAGES[target_name],
        )

        save_record(
            text,
            translated,
            source_name,
            target_name,
        )

        st.session_state.last_result = {

            "text": translated,

            "target": target_name,

            "pdf": create_pdf(
                translated,
                target_name,
            ),

            "docx": create_docx(
                translated
            ),
        }

    result = st.session_state.get(
        "last_result"
    )

    if result:

        st.subheader(
            "✅ Translated Text"
        )

        st.text_area(
            "Output",
            result["text"],
            height=300,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.download_button(
                "📄 Download TXT",
                result["text"],
                file_name=(
                    f"translated_"
                    f"{result['target']}.txt"
                ),
                mime="text/plain",
                use_container_width=True,
            )

        with col2:

            if result["pdf"]:

                st.download_button(
                    "📕 Download PDF",
                    result["pdf"],
                    file_name=(
                        f"translated_"
                        f"{result['target']}.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                )

            else:

                st.warning(
                    "PDF unavailable. "
                    "Download the required Noto font first."
                )

        with col3:

            st.download_button(
                "📝 Download DOCX",
                result["docx"],
                file_name=(
                    f"translated_"
                    f"{result['target']}.docx"
                ),
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.wordprocessingml.document"
                ),
                use_container_width=True,
            )


# =========================================================
# HISTORY
# =========================================================

if page == "📚 History":

    st.title(
        "📚 Translation History"
    )

    search = st.text_input(
        "🔎 Search translation history",
        placeholder="Search saved translations...",
    )

    records = load_records(search)

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Records",
        collection.count(),
    )

    col2.metric(
        "Shown",
        len(records),
    )

    st.divider()

    if not records:

        st.info(
            "No translation records found."
        )

    for record in records:

        with st.container(
            border=True
        ):

            head, stamp = st.columns(
                [3, 1]
            )

            head.markdown(
                f"### 🌐 "
                f"{record.get('source_language', '?')}"
                f" → "
                f"{record.get('target_language', '?')}"
            )

            stamp.caption(
                record.get(
                    "timestamp",
                    ""
                )
            )

            left, right = st.columns(2)

            with left:

                st.markdown(
                    "**Original Text**"
                )

                st.write(
                    record.get(
                        "source_text",
                        "",
                    )
                )

            with right:

                st.markdown(
                    "**Translated Text**"
                )

                st.write(
                    record["translated"]
                )

            st.caption(
                "🧵 Thread ID: "
                + record.get(
                    "thread_id",
                    "",
                )
            )

            with st.expander(
                "✏️ Edit / 🗑️ Delete"
            ):

                new_text = st.text_area(
                    "Edit translated text",
                    record["translated"],
                    key=f"edit_{record['id']}",
                    height=150,
                )

                c1, c2 = st.columns(2)

                if c1.button(
                    "💾 Save changes",
                    key=f"save_{record['id']}",
                ):

                    collection.update(
                        ids=[record["id"]],
                        documents=[new_text],
                    )

                    st.success(
                        "Record updated."
                    )

                    st.rerun()

                if c2.button(
                    "🗑️ Delete record",
                    key=f"delete_{record['id']}",
                ):

                    collection.delete(
                        ids=[record["id"]]
                    )

                    st.success(
                        "Record deleted."
                    )

                    st.rerun()

    if collection.count() > 0:

        st.divider()

        buffer = io.StringIO()

        writer = csv.writer(
            buffer
        )

        writer.writerow(
            [
                "timestamp",
                "source_language",
                "target_language",
                "source_text",
                "translated_text",
                "thread_id",
            ]
        )

        for record in load_records():

            writer.writerow(
                [
                    record.get(
                        "timestamp",
                        ""
                    ),
                    record.get(
                        "source_language",
                        ""
                    ),
                    record.get(
                        "target_language",
                        ""
                    ),
                    record.get(
                        "source_text",
                        ""
                    ),
                    record["translated"],
                    record.get(
                        "thread_id",
                        ""
                    ),
                ]
            )

        st.download_button(
            "⬇️ Export history (CSV)",
            buffer.getvalue().encode(
                "utf-8-sig"
            ),
            file_name="translation_history.csv",
            mime="text/csv",
        )

        confirm = st.checkbox(
            "I understand this will permanently delete all records"
        )

        if st.button(
            "🗑️ Clear All History",
            disabled=not confirm,
        ):

            all_ids = collection.get()["ids"]

            if all_ids:
                collection.delete(
                    ids=all_ids
                )

            st.success(
                "Translation history cleared."
            )

            st.rerun()


# =========================================================
# USAGE DASHBOARD
# =========================================================

if page == "📊 Usage":

    st.title(
        "📊 Usage Dashboard"
    )

    st.caption(
        "See visually how the translator is being used"
    )

    records = load_records()

    if not records:

        st.info(
            "No usage data yet. "
            "Translate something and it will appear here."
        )

    else:

        df = pd.DataFrame(records)

        for column in [
            "timestamp",
            "source_text",
            "source_language",
            "target_language",
            "thread_id",
        ]:

            if column not in df:
                df[column] = ""

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df["date"] = df[
            "timestamp"
        ].dt.date

        df["characters"] = (
            df["source_text"]
            .astype(str)
            .str.len()
        )

        df["pair"] = (
            df["source_language"]
            + " → "
            + df["target_language"]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Translations",
            len(df)
        )

        c2.metric(
            "Characters translated",
            f"{int(df['characters'].sum()):,}",
        )

        c3.metric(
            "Conversations",
            df["thread_id"].nunique(),
        )

        c4.metric(
            "Top target language",
            df[
                "target_language"
            ].mode().iat[0],
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Translations per day"
            )

            st.bar_chart(
                df.groupby("date").size()
            )

        with right:

            st.subheader(
                "Target languages"
            )

            st.bar_chart(
                df[
                    "target_language"
                ].value_counts()
            )

        st.subheader(
            "Language pairs"
        )

        pairs = (
            df["pair"]
            .value_counts()
            .rename_axis("Language pair")
            .reset_index(
                name="Translations"
            )
        )

        pairs["Share"] = (
            pairs["Translations"]
            / pairs["Translations"].sum()
            * 100
        )

        st.dataframe(
            pairs,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Share":
                st.column_config.ProgressColumn(
                    "Share",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )


# =========================================================
# REQUIREMENTS & GUIDE
# =========================================================

if page == "ℹ️ Requirements & Guide":

    st.title(
        "ℹ️ Requirements & Guide"
    )

    st.caption(
        "Check application requirements"
    )

    def pkg_version(name):

        try:

            return md.version(name)

        except md.PackageNotFoundError:

            return None

    packages = [

        (
            "streamlit",
            "Web interface"
        ),

        (
            "torch",
            "Runs translation model"
        ),

        (
            "transformers",
            "Loads NLLB-200"
        ),

        (
            "sentencepiece",
            "Tokenizer"
        ),

        (
            "chromadb",
            "Translation history"
        ),

        (
            "pypdf",
            "Reads PDF"
        ),

        (
            "python-docx",
            "Reads and creates DOCX"
        ),

        (
            "fpdf2",
            "Creates PDF"
        ),

        (
            "uharfbuzz",
            "Shapes Indian scripts"
        ),

        (
            "pandas",
            "Usage dashboard"
        ),
    ]

    package_rows = []

    for name, purpose in packages:

        version = pkg_version(name)

        package_rows.append(
            {
                "Package": name,
                "Used for": purpose,
                "Status":
                    "✅ Installed"
                    if version
                    else "❌ Missing",
                "Version":
                    version or "-",
            }
        )

    font_files = [
        (
            "Latin",
            LATIN_FONT
        )
    ] + [
        (
            language,
            BASE_DIR / filename
        )
        for language, filename
        in SCRIPT_FONTS.items()
    ]

    font_rows = [

        {
            "Language": language,
            "Font file": path.name,
            "Status":
                "✅ Found"
                if path.exists()
                else "❌ Missing",
        }

        for language, path
        in font_files
    ]

    ready = sum(
        row["Status"].startswith("✅")
        for row
        in package_rows + font_rows
    )

    total = (
        len(package_rows)
        + len(font_rows)
    )

    st.subheader(
        "Readiness"
    )

    st.progress(
        ready / total,
        text=(
            f"{ready} of {total} "
            "requirements ready"
        ),
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Device",
        "GPU (CUDA)"
        if DEVICE == "cuda"
        else "CPU",
    )

    c2.metric(
        "Model",
        MODEL.split("/")[-1]
    )

    c3.metric(
        "History records",
        collection.count()
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Python packages"
        )

        st.dataframe(
            pd.DataFrame(package_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.code(
            "pip install streamlit torch "
            "transformers sentencepiece "
            "chromadb pypdf python-docx "
            "fpdf2 uharfbuzz pandas",
            language="bash",
        )

    with right:

        st.subheader(
            "PDF Fonts"
        )

        st.dataframe(
            pd.DataFrame(font_rows),
            hide_index=True,
            use_container_width=True,
        )

        missing_fonts = [
            row
            for row in font_rows
            if row["Status"].startswith("❌")
        ]

        if st.button(
            "⬇️ Download missing fonts",
            type="primary",
            disabled=not missing_fonts,
            use_container_width=True,
        ):

            with st.spinner(
                "Downloading fonts..."
            ):

                st.session_state.font_results = (
                    download_fonts()
                )

            st.rerun()

        for (
            filename,
            success,
            message,
        ) in st.session_state.pop(
            "font_results",
            [],
        ):

            if success:

                st.success(
                    f"{filename}: {message}"
                )

            else:

                st.error(
                    f"{filename}: {message}"
                )

    st.divider()

    st.subheader(
        "How to use"
    )

    steps = [

        (
            "1️⃣ Choose languages",
            "Select source and target language."
        ),

        (
            "2️⃣ Add content",
            "Type text or upload TXT, PDF or DOCX."
        ),

        (
            "3️⃣ Translate",
            "Click Translate."
        ),

        (
            "4️⃣ Download",
            "Download TXT, PDF or DOCX."
        ),

    ]

    columns = st.columns(4)

    for column, (title, description) in zip(
        columns,
        steps,
    ):

        with column:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                st.write(
                    description
                )

    st.subheader(
        "Supported languages"
    )

    st.write(
        ", ".join(
            LANGUAGES.keys()
        )
    )

    st.info(
        "PDF files containing scanned images "
        "may require OCR because pypdf can only "
        "extract existing PDF text."
    )