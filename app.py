import io, uuid, torch, chromadb, streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document

MODEL = "facebook/nllb-200-distilled-600M"
LANG = {"English":"eng_Latn","Tamil":"tam_Taml","Telugu":"tel_Telu","Hindi":"hin_Deva",
        "Kannada":"kan_Knda","Malayalam":"mal_Mlym","French":"fra_Latn",
        "German":"deu_Latn","Spanish":"spa_Latn"}

st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.title{font-size:38px;font-weight:700}
.sub{color:#777;margin-bottom:25px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading NLLB model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    model.eval()
    return tok, model, device

@st.cache_resource
def get_db():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("translation_history")

def get_thread():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id

def read_file(file):
    data = file.getvalue()
    ext = file.name.lower().split(".")[-1]
    if ext == "txt":
        return data.decode("utf-8", errors="ignore")
    if ext == "pdf":
        r = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in r.pages)
    if ext == "docx":
        d = Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    return ""

def translate(text, source, target):
    tok, model, device = load_model()
    tok.src_lang = source
    target_id = tok.convert_tokens_to_ids(target)
    chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
    results, bar = [], st.progress(0)
    batch_size = 4 if device == "cuda" else 1

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False

            )
        results.extend(tok.batch_decode(output, skip_special_tokens=True))
        bar.progress(min((start+len(batch))/len(chunks), 1.0))
    bar.empty()
    return "\n\n".join(results)

def save_history(source, target, original, translated):
    get_db().add(
        ids=[str(uuid.uuid4())],
        documents=[translated],
        metadatas=[{
            "source": source,
            "target": target,
            "original": original,
            "thread": get_thread(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]
    )

def show_history(search=""):
    db = get_db()
    if db.count() == 0:
        st.info("📝 No previous translations yet.")
        return

    data = db.get(include=["documents", "metadatas"])
    records = list(zip(data.get("documents", []), data.get("metadatas", [])))
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
        thread = meta.get("thread", "Previous Session")
        timestamp = meta.get("time", "Unknown")

        with st.expander(f"🌐 {source} → {target} • {timestamp}"):
            st.caption(f"🧵 Thread ID: `{thread}`")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Original**")
                st.text_area("Original text", original, height=150, key=f"o{i}")

            with c2:
                st.markdown("**Translation**")
                st.text_area("Translated text", translated, height=150, key=f"t{i}")

            st.download_button(
                "📥 Download", translated,
                f"translation_{i}.txt", "text/plain", key=f"d{i}"
            )

# ================= UI =================

st.markdown('<div class="title">🌐 AI Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Fast multilingual translation with NLLB-200</div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
source_name = c1.selectbox("Source Language", list(LANG))
target_name = c2.selectbox("Target Language", list(LANG), index=1)

text = st.text_area(
    "Text to translate",
    height=160,
    placeholder="Type or paste your text here..."
)

file = st.file_uploader(
    "📁 Upload TXT / PDF / DOCX",
    type=["txt", "pdf", "docx"]
)

if file:
    text = read_file(file)
    if text:
        st.success(f"📄 {file.name} loaded successfully")

if st.button("🚀 Translate", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter text or upload a document.")
    elif source_name == target_name:
        st.warning("Please select different languages.")
    else:
        with st.spinner("Translating..."):
            result = translate(text, LANG[source_name], LANG[target_name])

        if result:
            save_history(source_name, target_name, text, result)
            st.success("✅ Translation completed successfully.")
            st.subheader("Translation")
            st.text_area("Result", result, height=300)
            st.download_button(
                "📥 Download Translation",
                result, "translation.txt", "text/plain",
                use_container_width=True
            )

with st.expander("🧵 Current Session"):
    c1, c2 = st.columns(2)
    c1.metric("History Records", get_db().count())
    c2.write(f"**Thread ID:** `{get_thread()}`")

st.divider()
st.subheader("🕘 Previous Translation History")

search = st.text_input(
    "🔍 Search History",
    placeholder="Search text, language or thread ID..."
)

c1, c2 = st.columns([3, 1])

with c1:
    show_history(search)

with c2:
    if st.button("🗑️ Clear All", use_container_width=True):
        ids = get_db().get()["ids"]
        if ids:
            get_db().delete(ids=ids)
        st.success("History cleared.")
        st.rerun()