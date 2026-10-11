import io, json, os, re, tempfile, time, urllib.request, uuid
from datetime import datetime

import ctranslate2
import streamlit as st
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

MODEL = "olob0/nllb-200-distilled-600M-ct2-int8_float16"
MAX_PAGES, CHUNK, BATCH, WINDOW, THREADS = 100, 400, 16, 32, 4

LANGS = {
    "English": "eng_Latn", "German": "deu_Latn", "Spanish": "spa_Latn", "Italian": "ita_Latn",
    "French": "fra_Latn", "Hindi": "hin_Deva", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Bengali": "ben_Beng", "Marathi": "mar_Deva",
    "Urdu": "urd_Arab", "Arabic": "arb_Arab", "Portuguese": "por_Latn", "Russian": "rus_Cyrl",
}

# PDF fonts: Latin is always loaded; other scripts get one extra font
NOTO = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{n}/hinted/ttf/{n}-Regular.ttf"
SCRIPT = {"Hindi": "NotoSansDevanagari", "Marathi": "NotoSansDevanagari", "Tamil": "NotoSansTamil",
          "Telugu": "NotoSansTelugu", "Kannada": "NotoSansKannada", "Malayalam": "NotoSansMalayalam",
          "Bengali": "NotoSansBengali", "Urdu": "NotoSansArabic", "Arabic": "NotoSansArabic"}

st.set_page_config(page_title="AIT Global Technologies", page_icon="🌍")
st.markdown("""<style>
.stApp{background:#fff}.stApp,.stApp p,.stApp label,.stApp span{color:#1B1B1B}
.t{background:#0B2A5B;border-radius:12px;padding:14px;text-align:center;font:700 1.3rem sans-serif}
.stButton>button,.stDownloadButton>button{background:#FFC107!important;color:#1B1B1B!important;border:0!important;font-weight:600!important}
</style><div class="t"><span style="color:#FFC107">AIT</span><span style="color:#fff"> GLOBAL TECHNOLOGIES</span></div>""",
            unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("thread", "THR-" + uuid.uuid4().hex[:8].upper())
ss.setdefault("result", None)
ss.setdefault("mem", [])

# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model .......")
def load_model():
    path = snapshot_download(MODEL)
    try:
        tok = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(path)
    return tok, ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=THREADS)

def translate(texts, src, tgt):
    tok, tr = load_model()
    tok.src_lang = src
    batch = [tok.convert_ids_to_tokens(i) for i in tok(texts, truncation=True, max_length=256).input_ids]
    out = []
    for r in tr.translate_batch(batch, target_prefix=[[tgt]] * len(batch), beam_size=1,
                                max_decoding_length=256, max_batch_size=2048, batch_type="tokens"):
        t = r.hypotheses[0]
        t = t[1:] if t and t[0] == tgt else t
        out.append(tok.decode(tok.convert_tokens_to_ids(t), skip_special_tokens=True))
    return out

# ---------------------------------------------------------------- helpers
def split(line):
    if len(line) <= CHUNK:
        return [line]
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?।؟])\s+", line):
        if cur and len(cur) + len(s) + 1 > CHUNK:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    return parts + ([cur] if cur else [])

def read(f):
    n, data = f.name.lower(), f.getvalue()
    if n.endswith(".pdf"):
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages[:MAX_PAGES])
    if n.endswith(".docx"):
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="replace")

@st.cache_resource
def font(name):
    path = os.path.join(tempfile.gettempdir(), f"{name}.ttf")
    if not os.path.exists(path) or os.path.getsize(path) < 10_000:
        req = urllib.request.Request(NOTO.format(n=name), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as o:
            o.write(r.read())
    return path

def make_pdf(lines, lang):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("base", "", font("NotoSans"))
    pdf.set_font("base", size=11)
    if lang in SCRIPT:
        pdf.add_font("extra", "", font(SCRIPT[lang]))
        pdf.set_fallback_fonts(["extra"])
        pdf.set_text_shaping(True)
    for l in lines:
        if l.strip():
            pdf.multi_cell(pdf.epw, 6, l, new_x="LMARGIN", new_y="NEXT", align="R" if lang in ("Urdu", "Arabic") else "L")
        else:
            pdf.ln(5)
    return bytes(pdf.output())

def make_docx(lines):
    d = Document()
    for l in lines:
        d.add_paragraph(l)
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()

# ---------------------------------------------------------------- history
@st.cache_resource
def col():
    c = chromadb.PersistentClient(path="chroma_db", settings=Settings(anonymized_telemetry=False))
    return c.get_or_create_collection("translations_v3")

def save(name, src, tgt, text):
    rid, meta = uuid.uuid4().hex, None
    meta = {"rid": rid, "thread": ss.thread, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name, "src": src, "tgt": tgt, "chars": len(text)}
    if not HAS_CHROMA:
        ss.mem.append({**meta, "text": text})
        return
    parts = [text[i:i + 1000] for i in range(0, len(text), 1000)] or [""]
    for s in range(0, len(parts), 100):
        p = parts[s:s + 100]
        col().add(ids=[f"{rid}-{s + j}" for j in range(len(p))], documents=p,
                  embeddings=[[0.0] * 8] * len(p),
                  metadatas=[{**meta, "i": s + j} for j in range(len(p))])

def find(thread, q):
    if not HAS_CHROMA:
        return [{**m, "doc": m["text"][:1000]} for m in reversed(ss.mem)
                if (not thread or m["thread"] == thread) and q.lower() in m["text"].lower()][:10]
    cond = [{"thread": thread}] if thread else []
    if not q:
        cond.append({"i": 0})
    where = {"$and": cond} if len(cond) > 1 else (cond[0] if cond else None)
    kw = {"where_document": {"$contains": q}} if q else {}
    r = col().get(where=where, limit=100, **kw)
    seen, out = set(), []
    for d, m in sorted(zip(r["documents"], r["metadatas"]), key=lambda x: x[1]["time"], reverse=True):
        if m["rid"] not in seen:
            seen.add(m["rid"])
            out.append({**m, "doc": d})
    return out[:10]

def full(rec):
    if not HAS_CHROMA:
        return rec["text"]
    r = col().get(where={"rid": rec["rid"]})
    return "".join(d for _, d in sorted(zip([m["i"] for m in r["metadatas"]], r["documents"])))

# ---------------------------------------------------------------- UI
c1, c2 = st.columns([3, 1])
c1.text_input("Thread ID", ss.thread, disabled=True)
c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
if c2.button("New thread"):
    ss.thread, ss.result = "THR-" + uuid.uuid4().hex[:8].upper(), None
    st.rerun()

names = list(LANGS)
a, b = st.columns(2)
src_name = a.selectbox("Translate from", names, index=names.index("English"))
tgt_name = b.selectbox("Translate to", names, index=names.index("Telugu"))

up = st.file_uploader("Upload a file (PDF: up to 100 pages)", type=["pdf", "txt", "docx", "json"])
typed = st.text_area("...or paste text", height=100)

st.subheader("Streaming Output")
box = st.empty()
box.text_area("Output", ss.result["text"] if ss.result else "", height=280, key="o0")

if st.button("Translate", type="primary"):
    if src_name == tgt_name:
        st.info("Source and target languages are the same.")
    elif not up and not typed.strip():
        st.warning("Upload a file or paste some text first.")
    else:
        text = read(up) if up else typed
        src, tgt = LANGS[src_name], LANGS[tgt_name]
        rows = [[c for c in split(l.strip())] if l.strip() else [] for l in text.splitlines()]
        todo = list(dict.fromkeys(c for r in rows for c in r))
        done, bar, t0 = {}, st.progress(0.0), time.time()

        def render():
            return "\n".join(" ".join(done.get(c, "") for c in r).strip() for r in rows)

        for w in range(0, len(todo), WINDOW):          # document order, window by window
            win = sorted(todo[w:w + WINDOW], key=len)  # sorted inside window = less padding
            for i in range(0, len(win), BATCH):
                for c, t in zip(win[i:i + BATCH], translate(win[i:i + BATCH], src, tgt)):
                    done[c] = t
            n = min(w + WINDOW, len(todo))
            left = int((len(todo) - n) / max(n / (time.time() - t0), 0.01))
            bar.progress(n / len(todo), text=f"{n}/{len(todo)} chunks · about {left // 60}m {left % 60}s left")
            box.text_area("Output", render(), height=280, key=f"o{w}")

        bar.empty()
        lines = render().split("\n")
        ss.result = {"text": "\n".join(lines), "lines": lines, "lang": tgt_name,
                     "json": {"source": src_name, "target": tgt_name, "translated_text": "\n".join(lines)}}
        try:
            save(up.name if up else "pasted text", src_name, tgt_name, ss.result["text"])
        except Exception as e:
            st.warning(f"Translated, but history was not saved: {e}")

r = ss.result
if r:
    st.subheader("Download")
    d = st.columns(4)
    if ss.get("pdf"):
        d[0].download_button("📄 PDF", ss.pdf, "translation.pdf", use_container_width=True)
    elif d[0].button("📄 Make PDF", use_container_width=True):
        ss.pdf = make_pdf(r["lines"], r["lang"])
        st.rerun()
    d[1].download_button("📝 TXT", r["text"].encode(), "translation.txt", use_container_width=True)
    d[2].download_button("📑 DOCX", make_docx(r["lines"]), "translation.docx", use_container_width=True)
    d[3].download_button("📋 JSON", json.dumps(r["json"], ensure_ascii=False, indent=2).encode(),
                         "translation.json", use_container_width=True)

st.divider()
st.subheader("Past records")
s1, s2 = st.columns([1, 2])
scope = s1.radio("Show", ["This thread", "All"], horizontal=True)
q = s2.text_input("Search", placeholder="Exact word...")
try:
    for rec in find(ss.thread if scope == "This thread" else None, q.strip()):
        with st.expander(f"{rec['time']} · {rec['name']} · {rec['src']} → {rec['tgt']}"):
            st.text(rec["doc"][:1000])
            st.caption(f"{rec['chars']:,} characters")
            if ss.get("prep") == rec["rid"]:
                st.download_button("⬇️ Download full", full(rec).encode(), f"{rec['name']}.txt", key=rec["rid"])
            elif st.button("Prepare download", key="p" + rec["rid"]):
                ss.prep = rec["rid"]
                st.rerun()
except Exception as e:
    st.warning(f"Could not load records: {e}")