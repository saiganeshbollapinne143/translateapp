import os
import uuid
import hmac
import hashlib
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
DB_PATH = str(BASE_DIR / "chroma_db")  # persistent (was /tmp, which gets wiped)

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
# AUTH  (admin = read + write, user = read + use translator)
# =========================================================
# Set passwords in .streamlit/secrets.toml:
#   ADMIN_PASSWORD = "your-strong-password"
#   USER_PASSWORD  = "another-password"
# or as environment variables. Defaults below are for local testing only.


def get_secret(key, default):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


USING_DEFAULT_PASSWORDS = (
    get_secret("ADMIN_PASSWORD", "") == "" or get_secret("USER_PASSWORD", "") == ""
)

USERS = {
    "admin": {
        "hash": sha(get_secret("ADMIN_PASSWORD", "admin123")),
        "role": "admin",
    },
    "user": {
        "hash": sha(get_secret("USER_PASSWORD", "user123")),
        "role": "user",
    },
}


def login_page():
    st.title("🔐 AI Translator Login")

    if USING_DEFAULT_PASSWORDS:
        st.warning("Default passwords are active. Set ADMIN_PASSWORD and USER_PASSWORD before deploying.")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        account = USERS.get(username.strip().lower())
        if account and hmac.compare_digest(account["hash"], sha(password)):
            st.session_state.auth = True
            st.session_state.username = username.strip().lower()
            st.session_state.role = account["role"]
            st.rerun()
        else:
            st.error("Invalid username or password.")


if not st.session_state.get("auth"):
    login_page()
    st.stop()

IS_ADMIN = st.session_state.role == "admin"

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
                "username": st.session_state.username,
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
        latin_name = None  # core Helvetica for French/German/Spanish/Italian

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
    """Return list of dicts (id, translated, metadata), newest first."""
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
st.sidebar.write(f"👤 **{st.session_state.username}**")
st.sidebar.caption(
    "Role: Admin (read + write)" if IS_ADMIN else "Role: User (read + use only)"
)

page = st.sidebar.radio("Navigation", ["🏠 Translator", "📚 History"])

st.sidebar.divider()
st.sidebar.write("**Current Thread**")
st.sidebar.code(st.session_state.thread_id)

if st.sidebar.button("🆕 New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

if st.sidebar.button("🚪 Logout"):
    for key in ["auth", "username", "role", "last_result"]:
        st.session_state.pop(key, None)
    st.rerun()

# =========================================================
# TRANSLATOR PAGE  (admin + user)
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

        # keep result in session so download buttons don't wipe it on rerun
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
# HISTORY PAGE  (admin: read + write, user: read only)
# =========================================================

if page == "📚 History":
    st.title("📚 Translation History")
    st.caption(
        "Full access: view, edit, delete, export."
        if IS_ADMIN
        else "Read-only view of saved translations."
    )

    search = st.text_input(
        "🔎 Search translation history", placeholder="Search saved translations..."
    )
    records = load_records(search)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", collection.count())
    col2.metric("Shown", len(records))
    col3.metric("Access", "Read + Write" if IS_ADMIN else "Read only")

    st.divider()

    if not records:
        st.info("No translation records found.")

    for rec in records:
        with st.container(border=True):
            head, stamp = st.columns([3, 1])
            head.markdown(
                f"### 🌐 {rec.get('source_language', '?')} → {rec.get('target_language', '?')}"
            )
            stamp.caption(
                f"{rec.get('timestamp', '')}  \n👤 {rec.get('username', 'unknown')}"
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**Original Text**")
                st.write(rec.get("source_text", ""))
            with right:
                st.markdown("**Translated Text**")
                st.write(rec["translated"])

            st.caption(f"🧵 Thread ID: {rec.get('thread_id', '')}")

            # ---------- WRITE ACTIONS: ADMIN ONLY ----------
            if IS_ADMIN:
                with st.expander("✏️ Edit / 🗑️ Delete (admin)"):
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

    # ---------- ADMIN-ONLY BULK ACTIONS ----------
    if IS_ADMIN and collection.count() > 0:
        st.divider()
        st.subheader("🛠️ Admin Tools")

        all_records = load_records()
        df = pd.DataFrame(all_records)
        st.download_button(
            "⬇️ Export history (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="translation_history.csv",
            mime="text/csv",
        )

        confirm = st.checkbox("I understand this will permanently delete all records")
        if st.button("🗑️ Clear All History", type="secondary", disabled=not confirm):
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            st.success("Translation history cleared.")
            st.rerun()