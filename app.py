import csv
import io
import importlib.metadata as md
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

st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

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

# Put these .ttf files next to app.py (download from Google Fonts)
LATIN_FONT = BASE_DIR / "NotoSans-Regular.ttf"
SCRIPT_FONTS = {
    "Tamil": "NotoSansTamil-Regular.ttf",
    "Hindi": "NotoSansDevanagari-Regular.ttf",
    "Telugu": "NotoSansTelugu-Regular.ttf",
    "Kannada": "NotoSansKannada-Regular.ttf",
    "Malayalam": "NotoSansMalayalam-Regular.ttf",
}

# =========================================================
# CHROMADB + MODEL
# =========================================================


@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name="translation_records")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


collection = load_database()
tokenizer, model = load_model()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# =========================================================
# HELPERS
# =========================================================


def extract_text(file):
    name = file.name.lower()
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(file).paragraphs)
    return ""


def split_text(text, size=300):
    chunks, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 <= size:
            current += (" " if current else "") + word
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


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
            chunk, return_tensors="pt", truncation=True, max_length=256
        ).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=2,
                do_sample=False,
            )

        results.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
        progress.progress((i + 1) / len(chunks))

    progress.empty()
    status.success("Translation completed!")
    return "\n\n".join(results)


def save_record(source_text, translated_text, source_language, target_language):
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[translated_text],
        metadatas=[
            {
                "source_text": source_text[:4000],
                "source_language": source_language,
                "target_language": target_language,
                "thread_id": st.session_state.thread_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
    )


def create_pdf(text, language):
    """Unicode PDF with proper shaping for Indian scripts. Returns bytes or None."""
    script_font = SCRIPT_FONTS.get(language)
    script_path = BASE_DIR / script_font if script_font else None

    if script_path and not script_path.exists():
        return None

    pdf = FPDF()
    pdf.add_page()

    if LATIN_FONT.exists():
        pdf.add_font("Latin", "", str(LATIN_FONT))
        latin_name = "Latin"
    else:
        if script_path:
            return None
        latin_name = None

    if script_path:
        pdf.add_font("Script", "", str(script_path))
        pdf.set_font("Script", size=12)
        pdf.set_fallback_fonts(["Latin"])
    elif latin_name:
        pdf.set_font("Latin", size=12)
    else:
        pdf.set_font("Helvetica", size=12)

    try:
        pdf.set_text_shaping(True)  # needs: pip install uharfbuzz
    except Exception:
        pass

    pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def load_records(search=""):
    """Return list of dicts (translated + metadata), newest first."""
    total = collection.count()
    if total == 0:
        return []

    if search:
        res = collection.query(query_texts=[search], n_results=min(20, total))
        ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
    else:
        res = collection.get()
        ids, docs, metas = res["ids"], res["documents"], res["metadatas"]

    records = [
        {"id": i, "translated": d, **(m or {})} for i, d, m in zip(ids, docs, metas)
    ]
    if not search:
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌐 AI Translator")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Translator", "📚 History", "📊 Usage", "ℹ️ Requirements & Guide"],
)

st.sidebar.divider()
st.sidebar.write("**Current Thread**")
st.sidebar.code(st.session_state.thread_id)

if st.sidebar.button("🆕 New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

# =========================================================
# TRANSLATOR PAGE
# =========================================================

if page == "🏠 Translator":
    st.title("🌐 AI Translator")
    st.caption("Translate text and documents using AI")

    col1, col2 = st.columns(2)
    with col1:
        source_name = st.selectbox("Source Language", list(LANGUAGES.keys()))
    with col2:
        target_name = st.selectbox("Target Language", list(LANGUAGES.keys()), index=1)

    text = st.text_area(
        "Enter text", height=180, placeholder="Type or paste your text here..."
    )
    file = st.file_uploader("Upload TXT, PDF or DOCX", type=["txt", "pdf", "docx"])

    if st.button("🔄 Translate", type="primary", use_container_width=True):
        if file:
            text = extract_text(file)

        if not text.strip():
            st.warning("Enter text or upload a document.")
            st.stop()

        if source_name == target_name:
            st.warning("Source and target languages are the same.")
            st.stop()

        translated = translate(text, LANGUAGES[source_name], LANGUAGES[target_name])
        save_record(text, translated, source_name, target_name)

        st.session_state.last_result = {
            "text": translated,
            "target": target_name,
            "pdf": create_pdf(translated, target_name),
        }

    result = st.session_state.get("last_result")
    if result:
        st.subheader("✅ Translated Text")
        st.text_area("Output", result["text"], height=300)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download TXT",
                result["text"],
                file_name=f"translated_{result['target']}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            if result["pdf"]:
                st.download_button(
                    "📕 Download PDF",
                    result["pdf"],
                    file_name=f"translated_{result['target']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.warning(
                    f"PDF unavailable: add the Noto font file(s) for {result['target']} next to app.py."
                )

# =========================================================
# HISTORY PAGE (read + write)
# =========================================================

if page == "📚 History":
    st.title("📚 Translation History")
    st.caption("View, edit and delete saved translation records")

    search = st.text_input(
        "🔎 Search translation history", placeholder="Search saved translations..."
    )
    records = load_records(search)

    col1, col2 = st.columns(2)
    col1.metric("Total Records", collection.count())
    col2.metric("Shown", len(records))

    st.divider()

    if not records:
        st.info("No translation records found.")

    for rec in records:
        with st.container(border=True):
            head, stamp = st.columns([3, 1])
            head.markdown(
                f"### 🌐 {rec.get('source_language', '?')} → {rec.get('target_language', '?')}"
            )
            stamp.caption(rec.get("timestamp", ""))

            left, right = st.columns(2)
            with left:
                st.markdown("**Original Text**")
                st.write(rec.get("source_text", ""))
            with right:
                st.markdown("**Translated Text**")
                st.write(rec["translated"])

            st.caption(f"🧵 Thread ID: {rec.get('thread_id', '')}")

            with st.expander("✏️ Edit / 🗑️ Delete"):
                new_text = st.text_area(
                    "Edit translated text",
                    rec["translated"],
                    key=f"edit_{rec['id']}",
                    height=150,
                )
                c1, c2 = st.columns(2)
                if c1.button("💾 Save changes", key=f"save_{rec['id']}"):
                    collection.update(ids=[rec["id"]], documents=[new_text])
                    st.success("Record updated.")
                    st.rerun()
                if c2.button("🗑️ Delete record", key=f"del_{rec['id']}"):
                    collection.delete(ids=[rec["id"]])
                    st.success("Record deleted.")
                    st.rerun()

    if collection.count() > 0:
        st.divider()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["timestamp", "source_language", "target_language", "source_text", "translated_text", "thread_id"]
        )
        for r in load_records():
            writer.writerow(
                [
                    r.get("timestamp", ""),
                    r.get("source_language", ""),
                    r.get("target_language", ""),
                    r.get("source_text", ""),
                    r["translated"],
                    r.get("thread_id", ""),
                ]
            )
        st.download_button(
            "⬇️ Export history (CSV)",
            buf.getvalue().encode("utf-8-sig"),
            file_name="translation_history.csv",
            mime="text/csv",
        )

        confirm = st.checkbox("I understand this will permanently delete all records")
        if st.button("🗑️ Clear All History", disabled=not confirm):
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            st.success("Translation history cleared.")
            st.rerun()


# =========================================================
# USAGE DASHBOARD (visual)
# =========================================================

if page == "📊 Usage":
    st.title("📊 Usage Dashboard")
    st.caption("See visually how the translator is being used")

    records = load_records()

    if not records:
        st.info("No usage data yet. Translate something and it will appear here.")
    else:
        df = pd.DataFrame(records)
        for col in ["timestamp", "source_text", "source_language", "target_language", "thread_id"]:
            if col not in df:
                df[col] = ""

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
        df["characters"] = df["source_text"].astype(str).str.len()
        df["pair"] = df["source_language"] + " → " + df["target_language"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Translations", len(df))
        c2.metric("Characters translated", f"{int(df['characters'].sum()):,}")
        c3.metric("Conversations", df["thread_id"].nunique())
        c4.metric("Top target language", df["target_language"].mode().iat[0])

        st.divider()

        left, right = st.columns(2)
        with left:
            st.subheader("Translations per day")
            st.bar_chart(df.groupby("date").size())
        with right:
            st.subheader("Target languages")
            st.bar_chart(df["target_language"].value_counts())

        st.subheader("Language pairs")
        pairs = (
            df["pair"]
            .value_counts()
            .rename_axis("Language pair")
            .reset_index(name="Translations")
        )
        pairs["Share"] = pairs["Translations"] / pairs["Translations"].sum() * 100
        st.dataframe(
            pairs,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Share": st.column_config.ProgressColumn(
                    "Share", format="%.0f%%", min_value=0, max_value=100
                )
            },
        )

        st.subheader("Recent activity")
        recent = (
            df.sort_values("timestamp", ascending=False)
            .head(10)[["timestamp", "pair", "characters", "thread_id"]]
            .rename(
                columns={
                    "timestamp": "Time",
                    "pair": "Language pair",
                    "characters": "Characters",
                    "thread_id": "Thread",
                }
            )
        )
        st.dataframe(recent, hide_index=True, use_container_width=True)
        st.caption("Character counts use the saved source text (stored up to 4,000 characters per record).")

# =========================================================
# REQUIREMENTS & GUIDE (visual)
# =========================================================

if page == "ℹ️ Requirements & Guide":
    st.title("ℹ️ Requirements & Guide")
    st.caption("What the app needs, whether it is ready, and how to use it")

    def pkg_version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    packages = [
        ("streamlit", "Web interface"),
        ("torch", "Runs the translation model"),
        ("transformers", "Loads NLLB-200"),
        ("sentencepiece", "Tokenizer for NLLB"),
        ("chromadb", "Stores translation history"),
        ("pypdf", "Reads PDF uploads"),
        ("python-docx", "Reads DOCX uploads"),
        ("fpdf2", "Creates PDF downloads"),
        ("uharfbuzz", "Shapes Tamil/Hindi text in PDFs"),
        ("pandas", "Usage charts"),
    ]
    pkg_rows = []
    for name, purpose in packages:
        version = pkg_version(name)
        pkg_rows.append(
            {
                "Package": name,
                "Used for": purpose,
                "Status": "✅ Installed" if version else "❌ Missing",
                "Version": version or "-",
            }
        )

    font_files = [("All languages (Latin text)", LATIN_FONT)] + [
        (lang, BASE_DIR / file) for lang, file in SCRIPT_FONTS.items()
    ]
    font_rows = [
        {
            "Language": lang,
            "Font file": path.name,
            "Status": "✅ Found" if path.exists() else "❌ Missing",
        }
        for lang, path in font_files
    ]

    ready = sum(r["Status"].startswith("✅") for r in pkg_rows + font_rows)
    total = len(pkg_rows) + len(font_rows)

    st.subheader("Readiness")
    st.progress(ready / total, text=f"{ready} of {total} requirements ready")

    c1, c2, c3 = st.columns(3)
    c1.metric("Device", "GPU (CUDA)" if DEVICE == "cuda" else "CPU")
    c2.metric("Model", MODEL.split("/")[-1])
    c3.metric("History records", collection.count())

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Python packages")
        st.dataframe(pd.DataFrame(pkg_rows), hide_index=True, use_container_width=True)
        st.code(
            "pip install streamlit torch transformers sentencepiece chromadb "
            "pypdf python-docx fpdf2 uharfbuzz pandas",
            language="bash",
        )
    with right:
        st.subheader("Fonts for PDF download")
        st.dataframe(pd.DataFrame(font_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Place the Noto .ttf files next to app.py. Missing fonts only affect the "
            "PDF download; translation and TXT download still work."
        )

    st.divider()
    st.subheader("How to use")

    steps = [
        ("1️⃣ Choose languages", "Pick the source and target language on the Translator page."),
        ("2️⃣ Add content", "Type or paste text, or upload a TXT, PDF or DOCX file."),
        ("3️⃣ Translate", "Click Translate, then download the result as TXT or PDF."),
        ("4️⃣ Review", "Open History to read, edit or delete records, and Usage to see charts."),
    ]
    for col, (title, desc) in zip(st.columns(4), steps):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(desc)

    st.subheader("Good to know")
    st.markdown(
        """
- Every translation is saved automatically to history (stored in the `chroma_db` folder).
- Long text is split into small chunks so nothing gets cut off.
- Supported languages: English, Hindi, Telugu, Tamil, Kannada, Malayalam, French, German, Spanish, Italian.
- Start the app with `streamlit run app.py`.
"""
    )