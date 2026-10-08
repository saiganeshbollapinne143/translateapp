import io, uuid, torch, chromadb, streamlit as st
from pathlib import Path
from functools import wraps
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document

MODEL = "facebook/nllb-200-distilled-600M"
CHROMA_DIR = Path("chroma_db")

LANG = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn"
}

st.set_page_config(
    page_title="AI Translator",
    page_icon="🌐",
    layout="wide"
)

# ---------- DECORATORS ----------
def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result
    return wrapper

def errors(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ {e}")
            return None
    return wrapper

# ---------- MODEL ----------
@st.cache_resource(show_spinner="Loading NLLB model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    model.to(device)
    model.eval()

    return tokenizer, model, device

# ---------- CHROMADB ----------
@st.cache_resource
def get_db():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )
    return client.get_or_create_collection(
        name="translation_history"
    )

def get_thread():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id

@errors
def save_history(source, target, original, translated):
    get_db().add(
        ids=[str(uuid.uuid4())],
        documents=[translated],
        metadatas=[{
            "source": source,
            "target": target,
            "original": original,
            "thread_id": get_thread(),
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }]
    )

# ---------- FILE ----------
@errors
def read_file(file):
    data = file.getvalue()
    ext = file.name.lower().split(".")[-1]

    if ext == "txt":
        return data.decode(
            "utf-8",
            errors="ignore"
        )

    if ext == "pdf":
        reader = PdfReader(
            io.BytesIO(data)
        )
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if ext == "docx":
        doc = Document(
            io.BytesIO(data)
        )
        return "\n".join(
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        )

    return ""

# ---------- TRANSLATE ----------
@timer
@errors
def translate(text, source, target):

    tokenizer, model, device = load_model()

    tokenizer.src_lang = source

    target_id = tokenizer.convert_tokens_to_ids(
        target
    )

    chunks = [
        text[i:i + 800]
        for i in range(
            0,
            len(text),
            800
        )
    ]

    results = []

    progress = st.progress(0)

    batch_size = 4 if device == "cuda" else 1

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
            max_length=512
        ).to(device)

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
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
                (start + len(batch))
                / len(chunks),
                1.0
            )
        )

    progress.empty()

    return "\n\n".join(results)

# ---------- HISTORY ----------
def show_history():

    db = get_db()

    if db.count() == 0:
        st.info("📝 No previous translations.")
        return

    data = db.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    for i, (doc, meta) in enumerate(
        reversed(
            list(
                zip(
                    data["documents"],
                    data["metadatas"]
                )
            )
        )
    ):

        with st.expander(
            f"🌐 {meta['source']} → "
            f"{meta['target']} | "
            f"{meta['time']}"
        ):

            st.caption(
                f"🧵 Thread: "
                f"{meta['thread_id']}"
            )

            st.text_area(
                "Original",
                meta["original"],
                height=100,
                key=f"original_{i}"
            )

            st.text_area(
                "Translation",
                doc,
                height=150,
                key=f"translation_{i}"
            )

            st.download_button(
                "📥 Download",
                doc,
                f"translation_{i}.txt",
                "text/plain",
                key=f"download_{i}"
            )

# ---------- UI ----------
st.title("🌐 AI Translator")

st.caption(
    "NLLB-200 • ChromaDB • "
    "Streamlit Cloud Optimized"
)

col1, col2 = st.columns(2)

source_name = col1.selectbox(
    "Source Language",
    list(LANG.keys())
)

target_name = col2.selectbox(
    "Target Language",
    list(LANG.keys()),
    index=1
)

text = st.text_area(
    "Enter text",
    height=160
)

file = st.file_uploader(
    "📁 Upload TXT / PDF / DOCX",
    type=[
        "txt",
        "pdf",
        "docx"
    ]
)

if file:
    text = read_file(file)

    if text:
        st.success(
            f"📄 Loaded {file.name}"
        )

if st.button(
    "🚀 Translate",
    type="primary",
    use_container_width=True
):

    if not text or not text.strip():

        st.warning(
            "Enter text or upload a file."
        )

    elif source_name == target_name:

        st.warning(
            "Select different languages."
        )

    else:

        result = translate(
            text,
            LANG[source_name],
            LANG[target_name]
        )

        if result:

            save_history(
                source_name,
                target_name,
                text,
                result
            )

            st.subheader(
                "✅ Translation"
            )

            st.text_area(
                "Output",
                result,
                height=300
            )

            st.download_button(
                "📥 Download Translation",
                result,
                "translation.txt",
                "text/plain",
                use_container_width=True
            )

            st.success(
                "Translation completed."
            )

# ---------- THREAD ----------
with st.expander(
    "🧵 Current Thread"
):

    st.write(
        f"Thread ID: `{get_thread()}`"
    )

    st.write(
        f"History records: `{get_db().count()}`"
    )

# ---------- HISTORY ----------
st.divider()

st.subheader(
    "🕘 Previous Translation History"
)

if st.button(
    "🗑️ Clear History"
):

    ids = get_db().get()["ids"]

    if ids:
        get_db().delete(ids=ids)

    st.success(
        "History cleared."
    )

    st.rerun()

show_history()