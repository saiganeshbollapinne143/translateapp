import io, uuid, torch, chromadb, streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document

# ================= CONFIG =================
MODEL = "facebook/nllb-200-distilled-600M"

LANG = {
    "English": "eng_Latn", "Tamil": "tam_Taml",
    "Telugu": "tel_Telu", "Hindi": "hin_Deva",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "French": "fra_Latn", "German": "deu_Latn",
    "Spanish": "spa_Latn"
}

st.set_page_config(
    page_title="AI Translator",
    page_icon="🌐",
    layout="wide"
)

# ================= CLEAN UI =================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    color: #777;
    margin-bottom: 25px;
}
.history-card {
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ================= MODEL =================
@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# ================= CHROMADB =================
@st.cache_resource
def get_db():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("translation_history")

# ================= THREAD =================
def get_thread():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id

# ================= FILE READER =================
def read_file(file):
    data = file.getvalue()
    ext = file.name.lower().split(".")[-1]

    if ext == "txt":
        return data.decode("utf-8", errors="ignore")

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if ext == "docx":
        doc = Document(io.BytesIO(data))
        return "\n".join(
            p.text for p in doc.paragraphs
            if p.text.strip()
        )

    return ""

# ================= TRANSLATION =================
def translate(text, source, target):

    tokenizer, model, device = load_model()

    tokenizer.src_lang = source
    target_id = tokenizer.convert_tokens_to_ids(target)

    chunks = [
        text[i:i + 1200]
        for i in range(0, len(text), 1200)
    ]

    results = []
    progress = st.progress(0)

    batch_size = 4 if device == "cuda" else 1

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

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
                max_new_tokens=512,
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
                (start + len(batch)) / len(chunks),
                1.0
            )
        )

    progress.empty()

    return "\n\n".join(results)

# ================= SAVE HISTORY =================
def save_history(
    source,
    target,
    original,
    translated
):

    get_db().add(
        ids=[str(uuid.uuid4())],
        documents=[translated],
        metadatas=[{
            "source": source,
            "target": target,
            "original": original,
            "thread": get_thread(),
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }]
    )

# ================= HISTORY =================
def show_history(search=""):

    db = get_db()

    if db.count() == 0:
        st.info("No previous translations yet.")
        return

    data = db.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    records = list(
        zip(
            data["documents"],
            data["metadatas"]
        )
    )

    records.reverse()

    if search:

        search = search.lower()

        records = [
            (doc, meta)
            for doc, meta in records
            if search in doc.lower()
            or search in meta["original"].lower()
            or search in meta["source"].lower()
            or search in meta["target"].lower()
            or search in meta["thread"].lower()
        ]

    if not records:
        st.warning("No matching history found.")
        return

    for i, (translated, meta) in enumerate(records):

        title = (
            f"{meta['source']} → "
            f"{meta['target']}   •   "
            f"{meta['time']}"
        )

        with st.expander(title):

            st.caption(
                f"🧵 Thread ID: `{meta['thread']}`"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original**")
                st.text_area(
                    "Original text",
                    meta["original"],
                    height=150,
                    key=f"original_{i}"
                )

            with col2:
                st.markdown("**Translation**")
                st.text_area(
                    "Translated text",
                    translated,
                    height=150,
                    key=f"translated_{i}"
                )

            st.download_button(
                "📥 Download",
                translated,
                f"translation_{i}.txt",
                "text/plain",
                key=f"download_{i}"
            )

# ================= HEADER =================
st.markdown(
    '<div class="main-title">🌐 AI Translator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fast multilingual translation powered by NLLB-200'
    '</div>',
    unsafe_allow_html=True
)

# ================= TRANSLATOR =================
left, right = st.columns(2)

source_name = left.selectbox(
    "Source Language",
    list(LANG.keys())
)

target_name = right.selectbox(
    "Target Language",
    list(LANG.keys()),
    index=1
)

text = st.text_area(
    "Text to translate",
    height=160,
    placeholder="Type or paste your text here..."
)

uploaded = st.file_uploader(
    "Upload document",
    type=["txt", "pdf", "docx"],
    help="Supports TXT, PDF and DOCX"
)

if uploaded:

    text = read_file(uploaded)

    if text:
        st.success(
            f"📄 {uploaded.name} loaded successfully"
        )

# ================= TRANSLATE =================
if st.button(
    "🚀 Translate",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter text or upload a document."
        )

    elif source_name == target_name:

        st.warning(
            "Please select different languages."
        )

    else:

        with st.spinner("Translating..."):

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

            st.success(
                "Translation completed successfully."
            )

            st.subheader("✅ Translation")

            st.text_area(
                "Result",
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

# ================= THREAD INFO =================
with st.expander("🧵 Current Session"):

    col1, col2 = st.columns(2)

    col1.metric(
        "History Records",
        get_db().count()
    )

    col2.write(
        f"**Thread ID:** `{get_thread()}`"
    )

# ================= HISTORY =================
st.divider()

st.subheader("🕘 Translation History")

search = st.text_input(
    "🔍 Search history",
    placeholder="Search by text, language or thread ID..."
)

c1, c2 = st.columns([3, 1])

with c1:
    if search:
        show_history(search)
    else:
        show_history()

with c2:
    if st.button(
        "🗑️ Clear All",
        use_container_width=True
    ):

        ids = get_db().get()["ids"]

        if ids:
            get_db().delete(ids=ids)

        st.success("History cleared.")
        st.rerun()