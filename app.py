# -----------------------------------------------------------------------------
# SQLITE PATCH FOR CHROMADB COMPATIBILITY (MUST BE AT VERY TOP)
# -----------------------------------------------------------------------------
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import io, os, re, json, uuid, time, html, zlib, shutil, textwrap
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="AI Translator", page_icon="🌍", layout="wide")

import chromadb
from transformers import AutoTokenizer
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -----------------------------------------------------------------------------
# SECRETS & CONFIGURATION
# -----------------------------------------------------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
COMPANY_NAME = st.secrets.get("COMPANY_NAME", "AIT GLOBAL")
MODEL_NAME = st.secrets.get("MODEL_NAME", "facebook/nllb-200-distilled-600M")
CT2_REPO = st.secrets.get("CT2_REPO", "")  # optional: pre-converted CTranslate2 repo on HF
CT2_DIR = os.path.join(os.getcwd(), "ct2_nllb_int8")
THREADS = int(st.secrets.get("TORCH_THREADS", os.cpu_count() or 2))
DEFAULT_BATCH_SIZE = int(st.secrets.get("DEFAULT_BATCH_SIZE", 16))
TM_LIMIT = 20000
CHROMA_COLLECTION = "translation_history_fast"
CHROMA_LEGACY = "translation_history"
QUALITY = {"Fast": 1, "Balanced": 2, "Best quality": 4}

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1100px; }}
.stButton>button {{ border-radius: 10px; font-weight: 600; }}
.banner {{
    background-color: #0B2D5C; padding: 14px 6px; border-radius: 10px;
    text-align: center; white-space: nowrap; font-size: clamp(12px, 2.5vw, 30px);
    font-weight: 800; margin: 15px 0 20px 0; width: 100%; box-sizing: border-box;
}}
.stream-box {{
    background-color: #ffffff; border: 1px solid #dcdcdc; border-radius: 8px;
    padding: 20px; font-family: 'Courier New', Courier, monospace; font-size: 13.5px;
    line-height: 1.5; white-space: pre-wrap; min-height: 250px; max-height: 500px;
    overflow-y: auto; color: #111111;
}}
</style>
<div class="banner">
    <span style="color:#FFD700">{html.escape(COMPANY_NAME)}</span>
    <span style="color:#FFFFFF"> TECHNOLOGIES - TRANSLATOR</span>
</div>
""",
    unsafe_allow_html=True,
)

LANGS = {
    "English": "eng_Latn", "Tamil": "tam_Taml", "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "French": "fra_Latn", "Spanish": "spa_Latn", "German": "deu_Latn", "Chinese": "zho_Hans",
    "Arabic": "arb_Arab", "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Portuguese": "por_Latn", "Russian": "rus_Cyrl", "Italian": "ita_Latn",
}

WIN = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONTS = {
    "Tamil": ["Nirmala.ttf"], "Hindi": ["Nirmala.ttf"], "Telugu": ["Nirmala.ttf"],
    "Chinese": ["msyh.ttc", "simsun.ttc"], "Japanese": ["YuGothR.ttc", "msgothic.ttc"],
    "Korean": ["malgun.ttf"], "Arabic": ["arial.ttf", "tahoma.ttf"],
}
DEFAULT_FONTS = ["arial.ttf", "calibri.ttf", "segoeui.ttf"]
PAGE_RE = re.compile(r"^(=+|-+)\s*(PAGE|PAGINA)\s*\d+.*", re.IGNORECASE)


# -----------------------------------------------------------------------------
# TRANSLATION BACKEND (CTranslate2 int8 if available, else PyTorch)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_backend():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    try:
        import ctranslate2

        if not (os.path.isdir(CT2_DIR) and os.listdir(CT2_DIR)):
            if CT2_REPO:
                from huggingface_hub import snapshot_download
                snapshot_download(CT2_REPO, local_dir=CT2_DIR, token=HF_TOKEN)
            else:
                from ctranslate2.converters import TransformersConverter
                tmp = CT2_DIR + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                TransformersConverter(MODEL_NAME, low_cpu_mem_usage=True).convert(
                    tmp, quantization="int8", force=True
                )
                os.replace(tmp, CT2_DIR)

        cuda = ctranslate2.get_cuda_device_count() > 0
        translator = ctranslate2.Translator(
            CT2_DIR,
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
            inter_threads=1,
            intra_threads=THREADS,
        )
        return {"kind": "ct2", "tok": tok, "model": translator}
    except Exception:
        shutil.rmtree(CT2_DIR + ".tmp", ignore_errors=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, low_cpu_mem_usage=True
    ).eval()
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device=device, dtype=torch.float16)
    else:
        device = "cpu"
        torch.set_num_threads(THREADS)
    return {"kind": "torch", "tok": tok, "model": model, "device": device}


def translate_chunk(sents, src, tgt, beam):
    be = load_backend()
    tok = be["tok"]
    tok.src_lang = LANGS[src]
    tgt_code = LANGS[tgt]

    if be["kind"] == "ct2":
        enc = [
            tok.convert_ids_to_tokens(tok(s, truncation=True, max_length=256)["input_ids"])
            for s in sents
        ]
        max_len = min(256, int(max(len(e) for e in enc) * 1.6) + 10)
        res = be["model"].translate_batch(
            enc,
            target_prefix=[[tgt_code]] * len(enc),
            beam_size=beam,
            max_decoding_length=max_len,
            max_batch_size=1024,
            batch_type="tokens",
        )
        return [
            tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True)
            for r in res
        ]

    import torch

    enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256).to(be["device"])
    max_new = min(256, int(enc["input_ids"].shape[1] * 1.6) + 10)
    with torch.inference_mode():
        ids = be["model"].generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
            num_beams=beam,
            max_new_tokens=max_new,
        )
    return tok.batch_decode(ids, skip_special_tokens=True)


def translate_stream(segments, src, tgt, chunk_size, beam, tr):
    """Fills `tr` (sentence -> translation) in place and yields (done, total)."""
    pending = [s for s in dict.fromkeys(segments) if s not in tr]
    total = len(pending)
    if not total:
        return
    load_backend()

    done = 0
    for w in range(0, total, chunk_size):
        win = sorted(pending[w : w + chunk_size], key=len)
        tr.update(zip(win, translate_chunk(win, src, tgt, beam)))
        done += len(win)
        yield done, total


# -----------------------------------------------------------------------------
# CHROMADB (lightweight hashed embeddings, no embedding model download)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


@st.cache_resource
def get_chroma_collection():
    client = get_chroma_client()
    try:
        return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
    except Exception:
        return client.get_or_create_collection(name=CHROMA_COLLECTION)


def embed(text, dim=128):
    v = [0.0] * dim
    for w in re.findall(r"\w+", text.lower())[:2000]:
        v[zlib.crc32(w.encode("utf-8")) % dim] += 1.0
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else [1.0] + [0.0] * (dim - 1)


def save_to_chroma(thread_id, source, target, original, translated, kind):
    try:
        get_chroma_collection().upsert(
            ids=[thread_id],
            documents=[translated],
            embeddings=[embed(translated)],
            metadatas=[{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_lang": source,
                "target_lang": target,
                "kind": kind,
                "original_text": original[:1000],
            }],
        )
    except Exception as e:
        st.warning(f"Could not persist record to ChromaDB: {e}")


def fetch_from_chroma(thread_id):
    colls = [get_chroma_collection()]
    try:
        colls.append(get_chroma_client().get_collection(CHROMA_LEGACY))
    except Exception:
        pass
    for c in colls:
        try:
            res = c.get(ids=[thread_id])
            if res and res["ids"]:
                return {"translated": res["documents"][0], "metadata": res["metadatas"][0]}
        except Exception:
            continue
    return None


def fetch_recent_records(limit=100):
    try:
        res = get_chroma_collection().get(include=["metadatas"], limit=limit)
        rows = list(zip(res["ids"], res["metadatas"]))
        rows.sort(key=lambda r: r[1].get("timestamp", ""), reverse=True)
        return rows
    except Exception:
        return []


# -----------------------------------------------------------------------------
# INPUT READING (returns ("text", str) or ("json", obj))
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=4)
def read_file_cached(name, data):
    name = name.lower()
    if name.endswith(".json"):
        return "json", json.loads(data.decode("utf-8-sig"))
    if name.endswith(".txt"):
        return "text", data.decode("utf-8-sig", errors="replace")
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [
            f"{'=' * 80} PAGE {i + 1}\n" + (p.extract_text() or "")
            for i, p in enumerate(reader.pages)
        ]
        return "text", "\n\n".join(pages)
    doc = Document(io.BytesIO(data))
    return "text", "\n".join(p.text for p in doc.paragraphs)


def read_prompt(raw):
    s = raw.strip()
    if s[:1] in "{[":
        try:
            return "json", json.loads(s)
        except ValueError:
            pass
    return "text", raw


# -----------------------------------------------------------------------------
# SEGMENTATION / PLANNING
# -----------------------------------------------------------------------------
def is_structural_line(line):
    s = line.strip()
    if not s:
        return True
    if set(s) <= {"-"} or set(s) <= {"="} or set(s) <= {"*"}:
        return True
    if PAGE_RE.match(s):
        return True
    return bool(re.match(r"^\d+[\.\)]?$", s))


def translatable(s):
    s = s.strip()
    if not s or not re.search(r"[^\W\d_]", s):
        return False
    if s.lower().startswith(("http://", "https://", "www.")):
        return False
    if " " not in s and re.search(r"[\d_/@:\\]", s):
        return False
    return True


def split_sentences(text, max_chars=400):
    out = []
    for s in re.split(r"(?<=[.!?।。！？؟])\s*", text):
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            out += textwrap.wrap(s, max_chars, break_long_words=False)
        else:
            out.append(s)
    return out


def plan_text(text):
    entries, segs = [], []
    for line in text.splitlines():
        if is_structural_line(line):
            entries.append(("keep", line))
            continue
        m = re.match(r"^(\s*(?:[-•*]\s+)?)(.*)", line)
        prefix, content = m.group(1), m.group(2)
        sents = split_sentences(content)
        entries.append(("tx", prefix, sents))
        segs += [s for s in sents if translatable(s)]

    def render(tr, strict=False):
        out = []
        for e in entries:
            if e[0] == "keep":
                out.append(e[1])
                continue
            _, prefix, sents = e
            if strict and any(translatable(s) and s not in tr for s in sents):
                break
            out.append(prefix + " ".join(tr.get(s, s) if translatable(s) else s for s in sents))
        return "\n".join(out)

    return segs, render, None


def plan_json(obj):
    segs = []

    def collect(o):
        if isinstance(o, str):
            segs.extend(s for s in split_sentences(o) if translatable(s))
        elif isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)

    def convert(o, tr):
        if isinstance(o, str):
            parts = split_sentences(o)
            if not parts:
                return o
            return " ".join(tr.get(s, s) if translatable(s) else s for s in parts)
        if isinstance(o, dict):
            return {k: convert(v, tr) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v, tr) for v in o]
        return o

    collect(obj)

    def to_obj(tr):
        return convert(obj, tr)

    def render(tr, strict=False):
        return json.dumps(to_obj(tr), ensure_ascii=False, indent=2)

    return segs, render, to_obj


# -----------------------------------------------------------------------------
# EXPORTS (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_registered_font(lang):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ] + [os.path.join(WIN, f) for f in FONTS.get(lang, []) + DEFAULT_FONTS]
    for path in candidates:
        if os.path.exists(path):
            try:
                name = f"F_{abs(hash(path))}"
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                pass
    return "Helvetica"


@st.cache_data(show_spinner=False, max_entries=8)
def make_pdf(text, lang):
    style = ParagraphStyle(
        "ClientTextFormat",
        parent=getSampleStyleSheet()["Normal"],
        fontName=get_registered_font(lang),
        fontSize=9.5,
        leading=13.5,
    )
    rtl = lang == "Arabic"
    if rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
        except ImportError:
            rtl = False

    story = []
    for line in text.splitlines():
        if PAGE_RE.match(line):
            story += [Spacer(1, 6), PageBreak()]
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if rtl and body:
            body = get_display(arabic_reshaper.reshape(body))
        safe = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = "&nbsp;" * indent + safe if safe else " "
        story += [Paragraph(safe, style), Spacer(1, 2)]

    out = io.BytesIO()
    SimpleDocTemplate(out, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36).build(story)
    return out.getvalue()


@st.cache_data(show_spinner=False, max_entries=8)
def make_docx(text):
    out, doc = io.BytesIO(), Document()
    for line in text.splitlines():
        if PAGE_RE.match(line):
            doc.add_page_break()
        doc.add_paragraph(line)
    doc.save(out)
    return out.getvalue()


def make_json(record):
    if record.get("obj") is not None:
        payload = record["obj"]
    else:
        payload = {
            "source_lang": record.get("source", ""),
            "target_lang": record["target"],
            "translated_text": record["translated"],
            "paragraphs": [l for l in record["translated"].splitlines() if l.strip()],
        }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("current_thread_id", None)
ss.setdefault("current_record", None)
ss.setdefault("tm", {})  # translation memory: (src, tgt, beam, sentence) -> translation

c1, c2 = st.columns(2)
source = c1.selectbox("Input language", list(LANGS), index=0)
target = c2.selectbox("Translate into", list(LANGS), index=1)
uploaded = st.file_uploader("Upload PDF, DOCX, TXT or JSON", type=["pdf", "txt", "docx", "json"])
prompt = st.text_area("Or enter text / JSON to translate", height=120)

o1, o2 = st.columns(2)
quality_name = o1.radio("Quality / speed", list(QUALITY), index=1, horizontal=True)
beam = QUALITY[quality_name]
chunk_opts = [8, 16, 32, 64]
chunk_size = o2.select_slider(
    "Streaming chunk (sentences per update)",
    chunk_opts,
    value=DEFAULT_BATCH_SIZE if DEFAULT_BATCH_SIZE in chunk_opts else 16,
)

if st.button("🚀 Translate", type="primary"):
    try:
        if uploaded:
            kind, payload = read_file_cached(uploaded.name, uploaded.getvalue())
        else:
            kind, payload = read_prompt(prompt)
    except Exception as e:
        st.error(f"Could not read input: {e}")
        st.stop()

    empty = (not payload) if kind == "json" else (not payload.strip())
    if empty:
        st.error("Upload a file or enter text first.")
    elif source == target:
        st.warning("Choose different input and output languages.")
    else:
        try:
            planner = plan_json if kind == "json" else plan_text
            segs, render, to_obj = planner(payload)
            tr = {
                s: ss.tm[(source, target, beam, s)]
                for s in set(segs)
                if (source, target, beam, s) in ss.tm
            }

            with st.spinner("Loading translation engine (first run converts the model once)..."):
                backend = load_backend()
            st.caption(f"Engine: {'CTranslate2 int8 (fast)' if backend['kind'] == 'ct2' else 'PyTorch (fallback)'}")

            bar = st.progress(0.0)
            box = st.empty()
            last = 0.0
            for done, total in translate_stream(segs, source, target, chunk_size, beam, tr):
                bar.progress(min(1.0, done / total))
                now = time.time()
                if now - last > 0.2:
                    last = now
                    box.markdown(
                        f'<div class="stream-box" dir="auto">{html.escape(render(tr, strict=True))}</div>',
                        unsafe_allow_html=True,
                    )
            bar.empty()

            final_text = render(tr)
            box.markdown(
                f'<div class="stream-box" dir="auto">{html.escape(final_text)}</div>',
                unsafe_allow_html=True,
            )

            if len(ss.tm) < TM_LIMIT:
                ss.tm.update({(source, target, beam, s): t for s, t in tr.items()})

            original = json.dumps(payload, ensure_ascii=False) if kind == "json" else payload
            thread_id = f"TR-{uuid.uuid4().hex[:10].upper()}"
            save_to_chroma(thread_id, source, target, original, final_text, kind)

            ss.current_thread_id = thread_id
            ss.current_record = {
                "translated": final_text,
                "target": target,
                "source": source,
                "kind": kind,
                "obj": to_obj(tr) if to_obj else None,
            }
            st.rerun()
        except Exception as e:
            st.error(f"Translation failed: {e}")

if ss.current_thread_id and ss.current_record:
    rec = ss.current_record
    st.success(f"Translation saved! System Thread ID: `{ss.current_thread_id}`")
    st.text_area("Final translation", rec["translated"], height=300)

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "⬇️ PDF", make_pdf(rec["translated"], rec["target"]), "translation.pdf", "application/pdf"
    )
    d2.download_button(
        "⬇️ DOCX",
        make_docx(rec["translated"]),
        "translation.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    d3.download_button("⬇️ TXT", rec["translated"], "translation.txt", "text/plain")
    d4.download_button("⬇️ JSON", make_json(rec), "translation.json", "application/json")

with st.expander("ChromaDB History / Search Thread ID"):
    tid_in = st.text_input("System Thread ID")
    if st.button("Load thread"):
        if tid_in.strip():
            record = fetch_from_chroma(tid_in.strip())
            if record:
                meta = record["metadata"]
                kind = meta.get("kind", "text")
                obj = None
                if kind == "json":
                    try:
                        obj = json.loads(record["translated"])
                    except ValueError:
                        kind = "text"
                ss.current_thread_id = tid_in.strip()
                ss.current_record = {
                    "translated": record["translated"],
                    "target": meta["target_lang"],
                    "source": meta.get("source_lang", ""),
                    "kind": kind,
                    "obj": obj,
                }
                st.rerun()
            else:
                st.warning("Thread ID not found in database.")

    st.subheader("Saved Records (latest 100)")
    rows = fetch_recent_records()
    if rows:
        for tid, meta in rows:
            st.write(
                f"`{tid}` — {meta.get('timestamp', 'N/A')} — "
                f"{meta.get('source_lang', '')} → {meta.get('target_lang', '')} "
                f"({meta.get('kind', 'text')})"
            )
    else:
        st.write("No saved translations in database yet.")