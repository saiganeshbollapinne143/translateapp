
import glob, io, json, os, re, uuid
from datetime import datetime
import ctranslate2
import streamlit as st
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import chromadb
from chromadb.config import Settings

MODEL_NAME = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
BATCH_SIZE, MAX_CHUNK_CHARS = 4, 400
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

LANGUAGES = {
    "English":"eng_Latn", "Hindi":"hin_Deva", "Telugu":"tel_Telu",
    "Tamil":"tam_Taml", "Kannada":"kan_Knda", "Malayalam":"mal_Mlym",
    "Bengali":"ben_Beng", "Marathi":"mar_Deva", "Urdu":"urd_Arab",
    "French":"fra_Latn", "German":"deu_Latn", "Spanish":"spa_Latn",
    "Italian":"ita_Latn", "Arabic":"arb_Arab",
    "Chinese (Simplified)":"zho_Hans", "Japanese":"jpn_Jpan"
}

st.set_page_config(page_title="AIT Global Technologies", page_icon="🌍", layout="centered")
st.markdown("""
<style>
.stApp,[data-testid="stHeader"]{background:#fff}
.stApp p,.stApp label,.stApp span,.stApp li{color:#1b1b1b}
.block-container{padding-top:2rem!important;max-width:820px}
.title-card{background:#0b2a5b;border-radius:12px;padding:18px 12px;
text-align:center;margin-bottom:20px}
.title-card h1{font-size:clamp(18px,4vw,26px);margin:0;white-space:normal}
.ait{color:#ffc107!important}.rest{color:white!important}
.stButton>button,.stDownloadButton>button{
background:#ffc107!important;color:#1b1b1b!important;
border:0!important;border-radius:8px!important;font-weight:600!important}
.stDownloadButton>button{width:100%}
</style>
<div class="title-card"><h1><span class="ait">AIT</span>
<span class="rest"> GLOBAL TECHNOLOGIES</span></h1></div>
""", unsafe_allow_html=True)

def new_thread_id():
    return "THR-" + uuid.uuid4().hex[:8].upper()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection("translations")

def save_record(thread, filename, src, tgt, source, translated):
    get_collection().add(
        ids=[uuid.uuid4().hex],
        documents=[translated],
        metadatas=[{
            "thread_id":thread,
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_name":filename,
            "source_lang":src,
            "target_lang":tgt,
            "source_preview":source[:500],
            "chars":len(translated)
        }]
    )

def load_records(thread=None, search=""):
    col = get_collection()
    where = {"thread_id":thread} if thread else None
    if search.strip() and col.count():
        res = col.query(
            query_texts=[search.strip()],
            n_results=min(20, col.count()),
            where=where
        )
        records = [
            {"doc":d,"meta":m}
            for d,m in zip(res["documents"][0],res["metadatas"][0])
        ]
    else:
        res = col.get(where=where, include=["documents","metadatas"])
        records = [
            {"doc":d,"meta":m}
            for d,m in zip(res["documents"],res["metadatas"])
        ]
        if search.strip():
            q = search.lower()
            records = [r for r in records if q in r["doc"].lower()
                       or q in r["meta"].get("source_preview","").lower()]
    return sorted(records,key=lambda r:r["meta"].get("timestamp",""),reverse=True)[:50]

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    path = snapshot_download(MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(path)
    model = ctranslate2.Translator(
        path,device="cpu",compute_type="int8",
        inter_threads=1,intra_threads=2
    )
    return tok,model

def translate_batch(texts,src,tgt):
    tok,model = load_model()
    tok.src_lang = src
    batch = [tok.convert_ids_to_tokens(
        tok(text,truncation=True,max_length=256).input_ids
    ) for text in texts]
    results = model.translate_batch(
        batch,target_prefix=[[tgt]]*len(batch),
        beam_size=2,max_decoding_length=300
    )
    output = []
    for r in results:
        tokens = r.hypotheses[0]
        if tokens and tokens[0] == tgt:
            tokens = tokens[1:]
        output.append(tok.decode(
            tok.convert_tokens_to_ids(tokens),skip_special_tokens=True
        ))
    return output

def split_long(line):
    if len(line) <= MAX_CHUNK_CHARS:
        return [line]
    sentences = re.split(r"(?<=[.!?।።؟])\s+",line)
    chunks,cur = [],""
    for s in sentences:
        if cur and len(cur)+len(s)+1 > MAX_CHUNK_CHARS:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks

def read_upload(f):
    data,name = f.getvalue(),f.name.lower()
    if name.endswith(".pdf"):
        return "txt","\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        return "txt","\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if name.endswith(".json"):
        return "json",json.loads(data.decode("utf-8",errors="replace"))
    return "txt",data.decode("utf-8",errors="replace")

def map_strings(obj,fn):
    if isinstance(obj,str):
        return fn(obj)
    if isinstance(obj,list):
        return [map_strings(x,fn) for x in obj]
    if isinstance(obj,dict):
        return {k:map_strings(v,fn) for k,v in obj.items()}
    return obj

def collect_strings(obj,acc):
    map_strings(obj,lambda s:acc.append(s) or s)
    return acc

def find_font():
    for pattern in [
        "font.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf","C:/Windows/Fonts/Nirmala.ttf"
    ]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return None

def make_pdf(lines):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True,margin=15)
    pdf.add_page()
    font = find_font()
    if font:
        pdf.add_font("U","",font)
        pdf.set_font("U",size=11)
    else:
        pdf.set_font("Helvetica",size=11)
    for line in lines:
        if not line.strip():
            pdf.ln(5)
        else:
            if not font:
                line = line.encode("latin-1","replace").decode("latin-1")
            pdf.multi_cell(0,6,line)
    return bytes(pdf.output())

def make_docx(lines):
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

# ----------------------------- INPUTS
a,b = st.columns([3,1])
a.text_input("System-generated Thread ID",st.session_state.thread_id,disabled=True)
if b.button("New thread"):
    st.session_state.thread_id = new_thread_id()
    st.session_state.pop("result",None)
    st.rerun()

c1,c2 = st.columns(2)
src_name = c1.selectbox("Translate from",list(LANGUAGES),index=0)
tgt_name = c2.selectbox("Translate to",list(LANGUAGES),index=1)
uploaded = st.file_uploader("Upload file",type=["pdf","docx","txt","json"])
typed = st.text_area("Or paste text",height=100)

# ----------------------------- STREAM TRANSLATION OUTPUT ONLY
if st.button("Translate",type="primary",use_container_width=True):
    if src_name == tgt_name:
        st.warning("Choose different source and target languages.")
    elif not uploaded and not typed.strip():
        st.warning("Upload a file or paste text first.")
    else:
        try:
            kind,content = read_upload(uploaded) if uploaded else ("txt",typed)
            source_text = json.dumps(content,ensure_ascii=False) if kind=="json" else content
            src,tgt = LANGUAGES[src_name],LANGUAGES[tgt_name]
            st.subheader("Translated Output")
            output_box = st.empty()
            progress = st.progress(0.0)
            done = {}

            if kind == "json":
                strings = collect_strings(content,[])
                unique = list(dict.fromkeys(s for s in strings if s.strip()))
                total = max(len(unique),1)
                for i in range(0,len(unique),BATCH_SIZE):
                    batch = unique[i:i+BATCH_SIZE]
                    done.update(zip(batch,translate_batch(batch,src,tgt)))
                    partial = map_strings(content,lambda s:done.get(s,s))
                    shown = json.dumps(partial,ensure_ascii=False,indent=2)
                    output_box.code(shown,language="json")
                    progress.progress(min((i+len(batch))/total,1.0))
                final_json = map_strings(content,lambda s:done.get(s,s))
                translated_text = json.dumps(final_json,ensure_ascii=False,indent=2)
                lines = translated_text.splitlines()
                result_json = final_json
            else:
                original_lines = content.splitlines()
                pieces = [split_long(x.strip()) if x.strip() else [] for x in original_lines]
                unique = list(dict.fromkeys(p for group in pieces for p in group if p.strip()))
                total = max(len(unique),1)
                for i in range(0,len(unique),BATCH_SIZE):
                    batch = unique[i:i+BATCH_SIZE]
                    done.update(zip(batch,translate_batch(batch,src,tgt)))
                    lines = [
                        " ".join(done.get(p,p) for p in group) if group else ""
                        for group in pieces
                    ]
                    # Only translated text is shown while batches finish
                    output_box.text_area("Translation","\n".join(lines),height=250)
                    progress.progress(min((i+len(batch))/total,1.0))
                translated_text = "\n".join(lines)
                result_json = {
                    "thread_id":st.session_state.thread_id,
                    "source_language":src_name,
                    "target_language":tgt_name,
                    "translated_text":translated_text
                }

            progress.empty()
            st.session_state.result = {"lines":lines,"json":result_json}
            save_record(
                st.session_state.thread_id,
                uploaded.name if uploaded else "pasted text",
                src_name,tgt_name,source_text,translated_text
            )
            st.success("Translation completed and history saved.")
        except Exception as e:
            st.error(f"Translation failed: {e}")

# ----------------------------- DOWNLOADS
result = st.session_state.get("result")
if result:
    st.markdown("**Download translation**")
    c1,c2,c3,c4 = st.columns(4)
    c1.download_button("PDF",make_pdf(result["lines"]),"translation.pdf","application/pdf")
    c2.download_button("TXT","\n".join(result["lines"]),"translation.txt","text/plain")
    c3.download_button(
        "DOCX",make_docx(result["lines"]),"translation.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    c4.download_button(
        "JSON",json.dumps(result["json"],ensure_ascii=False,indent=2),
        "translation.json","application/json"
    )
    if not find_font():
        st.caption("No Unicode font found: non-Latin characters in PDF may display incorrectly.")

# ----------------------------- CHROMADB HISTORY
st.divider()
st.subheader("Past Translation History")
h1,h2 = st.columns([1,2])
scope = h1.radio("Show",["This thread","All threads"],horizontal=True)
search = h2.text_input("Search history",placeholder="Search translated text")
try:
    records = load_records(
        st.session_state.thread_id if scope=="This thread" else None,search
    )
    if not records:
        st.caption("No saved records yet.")
    for r in records:
        m = r["meta"]
        title = f'{m.get("timestamp","")} · {m.get("source_name","")} · {m.get("source_lang","")} → {m.get("target_lang","")} · {m.get("thread_id","")}'
        with st.expander(title):
            st.caption("Source preview")
            st.text(m.get("source_preview",""))
            st.caption("Saved translation")
            st.text(r["doc"])
except Exception as e:
    st.warning(f"Could not load ChromaDB history: {e}")
