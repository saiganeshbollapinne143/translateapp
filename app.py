import os, io, uuid, gc, torch, chromadb, streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document

# ================= CONFIG =================
st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

if "HF_TOKEN" not in os.environ and "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

MODEL = "facebook/nllb-200-distilled-600M"

LANG = {
    "English": "eng_Latn", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Hindi": "hin_Deva", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "French": "fra_Latn", "German": "deu_Latn", "Spanish": "spa_Latn"
}

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.title{font-size:38px;font-weight:700}
.sub{color:#777;margin-bottom:25px}
</style>
""", unsafe_allow_html=True)

# ================= THREAD =================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ================= MODEL =================
@st.cache_resource(show_spinner="Loading NLLB model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model, device

# ================= CHROMADB =================
@st.cache_resource
def get_db():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("translation_history")

# ================= FILE READER =================
def read_file(file):
    data = file.getvalue()
    ext = file.name.lower().split(".")[-1]

    if ext == "txt":
        return data.decode("utf-8", errors="ignore")

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    return ""

# ================= TRANSLATE =================
def translate(text, source, target):
    tokenizer, model, device = load_model()

    tokenizer.src_lang = source
    target_id = tokenizer.lang_code_to_id[target]

    chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
    results = []
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False
            )

        results.append(
            tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        )

        progress.progress((i + 1) / len(chunks))

    progress.empty()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return " ".join(results)

# ================= SAVE HISTORY =================
def save_history(source, target, original, translated):
    get_db().add(
        ids=[str(uuid.uuid4())],
        documents=[translated],
        metadatas=[{
            "source": source,
            "target": target,
            "original": original,
            "thread": st.session_state.thread_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]
    )

# ================= HISTORY =================
def show_history(search=""):
    db = get_db()

    if db.count() == 0:
        st.info("📝 No previous translations.")
        return

    data = db.get(include=["documents", "metadatas"])
    records = list(zip(
        data.get("documents", []),
        data.get("metadatas", [])
    ))
    records.reverse()

    if search:
        search = search.lower()
        records = [
            (doc, meta) for doc, meta in records
            if search in str(doc).lower()
            or search in str(meta.get("original", "")).lower()
            or search in str(meta.get("source", "")).lower()
            or search in str(meta.get("target", "")).lower()
            or search in str(meta.get("thread", "")).lower()
        ]

    if not records:
        st.warning("No matching history found.")
        return

    for i, (translated, meta) in enumerate(records):
        source = meta.get("source", "Unknown")
        target = meta.get("target", "Unknown")
        original = meta.get("original", "")
        thread = meta.get("thread", "Legacy")
        time = meta.get("time", "Unknown")

        with st.expander(
            f"🌐 {source} → {target} | Thread: {thread} | {time}"
        ):
            st.caption(f"🧵 Thread ID: `{thread}`")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Original**")
                st.text_area(
                    "Original",
                    original,
                    height=150,
                    key=f"original_{i}"
                )

            with c2:
                st.markdown("**Translation**")
                st.text_area(
                    "Translation",
                    translated,
                    height=150,
                    key=f"translation_{i}"
                )

            st.download_button(
                "📥 Download",
                translated,
                f"translation_{i}.txt",
                "text/plain",
                key=f"download_{i}"
            )

# ================= UI =================
st.markdown(
    '<div class="title">🌐 AI Translator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub">Fast multilingual translation powered by NLLB-200</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

source_name = c1.selectbox(
    "Source Language",
    list(LANG.keys())
)

target_name = c2.selectbox(
    "Target Language",
    list(LANG.keys()),
    index=1
)

input_text = st.text_area(
    "Text to translate",
    height=160,
    placeholder="Type or paste text here..."
)

uploaded = st.file_uploader(
    "📁 Upload TXT / PDF / DOCX",
    type=["txt", "pdf", "docx"]
)

if uploaded:
    st.success(f"📄 {uploaded.name} loaded successfully")

# ================= TRANSLATE BUTTON =================
if st.button(
    "🚀 Translate",
    type="primary",
    use_container_width=True
):
    source_text = read_file(uploaded) if uploaded else input_text

    if not source_text.strip():
        st.warning("Please enter text or upload a valid document.")

    elif source_name == target_name:
        st.warning("Please select different source and target languages.")

    else:
        try:
            with st.spinner("Translating..."):
                result = translate(
                    source_text,
                    LANG[source_name],
                    LANG[target_name]
                )

            save_history(
                source_name,
                target_name,
                source_text,
                result
            )

            st.success("✅ Translation completed.")

            st.subheader("Translation Result")

            st.text_area(
                "Result Output",
                result,
                height=250
            )

            st.download_button(
                "📥 Download Translation",
                result,
                "translation.txt",
                "text/plain",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Translation Error: {e}")

# ================= THREAD =================
st.divider()

c1, c2 = st.columns([4, 1])

with c1:
    st.write("🧵 **Current Thread ID**")
    st.code(st.session_state.thread_id)

with c2:
    if st.button(
        "＋ New Thread",
        use_container_width=True
    ):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ================= HISTORY =================
st.divider()

st.subheader("🕘 Translation History")

search = st.text_input(
    "🔍 Search History",
    placeholder="Search text, language or thread ID..."
)

show_history(search)

# ================= CLEAR HISTORY =================
if st.button("🗑️ Clear All History"):
    ids = get_db().get()["ids"]

    if ids:
        get_db().delete(ids=ids)

    st.success("History cleared.")
    st.rerun()