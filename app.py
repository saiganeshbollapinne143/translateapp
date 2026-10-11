import io, json, re, uuid, time, hashlib, importlib
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

MODEL = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
LANG = {"English":"eng_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Tamil":"tam_Taml","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Urdu":"urd_Arab","French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn","Italian":"ita_Latn","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan"}

st.set_page_config(page_title="AIT GLOBAL TRANSLATOR", page_icon="🌐", layout="centered")
st.markdown("""<style>
.stApp{background:#f4f7fc;color:#18243b}.block-container{max-width:900px;padding:1.5rem 1rem}
.hero{background:linear-gradient(120deg,#071d43,#124b89);padding:24px;border-radius:16px;text-align:center;margin-bottom:18px}
.hero h1{color:white;font-size:30px;margin:0}.hero span{color:#ffc928}.hero p{color:#e0eaff}
.stButton>button,.stDownloadButton>button{background:#ffc928;color:#10213d;border:0;border-radius:9px;font-weight:700;min-height:42px}
.live{background:white;border:1px solid #d7e2f3;border-left:5px solid #124b89;padding:14px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere}
</style><div class="hero"><h1><span>AIT GLOBAL</span> TRANSLATOR</h1><p>Multilingual 40+ Page Documents · Incremental Caching · Live Chunk Output</p></div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading translation model...")
def load():
    ct = importlib.import_module("ctranslate2")
    path = snapshot_download(MODEL)
    return AutoTokenizer.from_pretrained(path), ct.Translator(path, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)

def read_file(f):
    b, n = f.getvalue(), f.name.lower()
    if n.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(b))
        text_pages = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                text_pages.append(f"--- Page {i+1} ---\n" + t)
        return "text", "\n\n".join(text_pages)
    if n.endswith(".docx"):
        return "text", "\n".join(p.text for p in Document(io.BytesIO(b)).paragraphs)
    if n.endswith(".json"):
        return "json", json.loads(b.decode("utf-8-sig"))
    return "text", b.decode("utf-8-sig", errors="replace")

def split_text(text, tok, limit=350):
    sentences = re.split(r'(?<=[.!?।؟])\s+|\n+', text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        words = sentence.split()
        pieces, part = [], ""
        for word in words:
            candidate = (part + " " + word).strip()
            if part and len(tok(candidate, add_special_tokens=True).input_ids) > limit:
                pieces.append(part)
                part = word
            else:
                part = candidate
        if part:
            pieces.append(part)
        for piece in pieces:
            candidate = (current + " " + piece).strip()
            if current and len(tok(candidate, add_special_tokens=True).input_ids) > limit:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks

def translate_chunk(text, src, tgt, tok, model, update):
    tok.src_lang = src
    ids = tok(text, add_special_tokens=True).input_ids
    source = tok.convert_ids_to_tokens(ids)
    generated = []
    def callback(step):
        if step.token != tgt:
            generated.append(step.token)
    def decode(tokens):
        tokens = [t for t in tokens if t != tgt]
        return tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True)
    try:
        future = model.translate_batch(
            [source], target_prefix=[[tgt]], beam_size=1,
            max_decoding_length=512, callback=callback, asynchronous=True
        )[0]
        last = -1
        while not future.done():
            if len(generated) != last:
                update(decode(generated) or "Generating…")
                last = len(generated)
            time.sleep(0.05)
        result = decode(future.result().hypotheses[0])
    except TypeError:
        result = decode(model.translate_batch(
            [source], target_prefix=[[tgt]], beam_size=2,
            max_decoding_length=512
        )[0].hypotheses[0])
    return result.strip()

def walk_json(obj, fn):
    if isinstance(obj, str):
        return fn(obj) if obj.strip() else obj
    if isinstance(obj, list):
        return [walk_json(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: walk_json(v, fn) for k, v in obj.items()}
    return obj

def docx_bytes(text):
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def pdf_bytes(text):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in text.splitlines():
        pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))
    return bytes(pdf.output())

if "thread" not in st.session_state:
    st.session_state.thread = "THR-" + uuid.uuid4().hex[:8].upper()
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "result" not in st.session_state:
    st.session_state.result = None

with st.container(border=True):
    left, right = st.columns([3, 1])
    left.text_input("Session ID", st.session_state.thread, disabled=True)
    if right.button("New session", use_container_width=True):
        st.session_state.thread = "THR-" + uuid.uuid4().hex[:8].upper()
        st.session_state.cache = {}
        st.session_state.result = None
        st.rerun()

    left, right = st.columns(2)
    src = left.selectbox("Source language", list(LANG))
    tgt = right.selectbox("Target language", list(LANG), index=2)
    file = st.file_uploader("Upload document (PDF, DOCX, TXT, JSON)", type=["pdf", "docx", "txt", "json"])
    pasted = st.text_area("Or paste text", height=110, placeholder="Enter text to translate…")
    st.caption("Optimized for large 40+ page documents with incremental chunk caching.")
    st.subheader("Live output")
    live = st.empty()
    progress = st.empty()

    if st.button("Translate document", type="primary", use_container_width=True):
        if src == tgt:
            st.warning("Choose different source and target languages.")
        elif not file and not pasted.strip():
            st.warning("Upload a document or enter text.")
        else:
            try:
                kind, data = read_file(file) if file else ("text", pasted)
                if kind == "text" and not data.strip():
                    st.error("No readable text found. Scanned PDFs may require OCR.")
                    st.stop()

                tok, model = load()
                
                if kind == "json":
                    strings = []
                    def gather(x):
                        if isinstance(x, str) and x.strip():
                            strings.append(x)
                        elif isinstance(x, dict):
                            for v in x.values(): gather(v)
                        elif isinstance(x, list):
                            for v in x: gather(v)
                    gather(data)
                    unique = list(dict.fromkeys(strings))
                    bar = st.progress(0, text="Preparing JSON translations…")
                    translated_map = {}
                    for i, value in enumerate(unique, 1):
                        key = hashlib.sha256((MODEL + src + tgt + value).encode("utf-8")).hexdigest()
                        if key in st.session_state.cache:
                            translated_map[value] = st.session_state.cache[key]
                        else:
                            def show(part, idx=i):
                                safe = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                live.markdown(f'<div class="live"><b>JSON Field {idx}/{len(unique)}</b><hr>{safe}</div>', unsafe_allow_html=True)
                            res = translate_chunk(value, LANG[src], LANG[tgt], tok, model, show)
                            st.session_state.cache[key] = res
                            translated_map[value] = res
                        bar.progress(i / max(len(unique), 1), text=f"Processed {i}/{len(unique)} JSON fields")
                    result = walk_json(data, lambda s: translated_map.get(s, s))
                    output = json.dumps(result, ensure_ascii=False, indent=2)
                    json_output = result
                else:
                    pieces = split_text(data, tok, limit=350)
                    if not pieces:
                        st.error("No readable text found.")
                        st.stop()
                    bar = st.progress(0, text=f"Processing {len(pieces)} text chunks (40-page mode)…")
                    translated = []
                    for i, piece in enumerate(pieces, 1):
                        key = hashlib.sha256((MODEL + src + tgt + piece).encode("utf-8")).hexdigest()
                        if key in st.session_state.cache:
                            value = st.session_state.cache[key]
                            live.markdown(f'<div class="live"><b>Chunk {i}/{len(pieces)} · Loaded from cache</b><hr>{value}</div>', unsafe_allow_html=True)
                        else:
                            def show(part, idx=i):
                                safe = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                live.markdown(f'<div class="live"><b>Chunk {idx}/{len(pieces)}</b><hr>{safe}</div>', unsafe_allow_html=True)
                            value = translate_chunk(piece, LANG[src], LANG[tgt], tok, model, show)
                            st.session_state.cache[key] = value
                        translated.append(value)
                        bar.progress(i / len(pieces), text=f"Translated chunk {i} of {len(pieces)}")
                    output = "\n\n".join(translated)
                    json_output = {"session_id": st.session_state.thread, "source_language": src, "target_language": tgt, "translated_text": output}

                st.session_state.result = {"text": output, "json": json_output}
                live.markdown('<div class="live"><b>Translation complete</b><hr>All 40 pages / chunks successfully translated and cached.</div>', unsafe_allow_html=True)
                st.success("Translation completed successfully.")
            except Exception as e:
                st.error(f"Translation failed: {e}")

r = st.session_state.result
if r:
    st.subheader("Download translated document")
    cols = st.columns(4)
    cols[0].download_button("📄 PDF", pdf_bytes(r["text"]), "translation.pdf", "application/pdf", use_container_width=True)
    cols[1].download_button("📝 TXT", r["text"].encode("utf-8"), "translation.txt", "text/plain", use_container_width=True)
    cols[2].download_button("📑 DOCX", docx_bytes(r["text"]), "translation.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    cols[3].download_button("📋 JSON", json.dumps(r["json"], ensure_ascii=False, indent=2).encode("utf-8"), "translation.json", "application/json", use_container_width=True)