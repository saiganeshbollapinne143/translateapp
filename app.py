import io, os, re, uuid, textwrap
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title="AI Translator", page_icon="🌍", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
h1 {background: linear-gradient(90deg,#2563eb,#7c3aed); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 800;}
.stButton>button {border-radius: 10px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)
st.title("🌍 AIT GLOBAL TECHNOLOGIES - TRANSLATOR")

LANGS = {
    "English": "eng_Latn", "Tamil": "tam_Taml", "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "French": "fra_Latn", "Spanish": "spa_Latn", "German": "deu_Latn", "Chinese": "zho_Hans",
    "Arabic": "arb_Arab", "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Portuguese": "por_Latn", "Russian": "rus_Cyrl", "Italian": "ita_Latn",
}

# Windows font candidates per language (for PDF output); falls back to Helvetica
WIN = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONTS = {
    "Tamil": ["Nirmala.ttf"], "Hindi": ["Nirmala.ttf"], "Telugu": ["Nirmala.ttf"],
    "Chinese": ["msyh.ttc", "simsun.ttc"], "Japanese": ["YuGothR.ttc", "msgothic.ttc"],
    "Korean": ["malgun.ttf"], "Arabic": ["arial.ttf", "tahoma.ttf"],
}
DEFAULT_FONTS = ["arial.ttf", "calibri.ttf", "segoeui.ttf"]


@st.cache_resource
def load_model():
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    if torch.cuda.is_available():
        device = "cuda"
        model = model.half().to(device)
    else:
        device = "cpu"
        torch.set_num_threads(os.cpu_count() or 4)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tok, model, device


def read_file(file):
    data, n = file.getvalue(), file.name.lower()
    if n.endswith(".txt"):
        return data.decode("utf-8-sig", errors="replace")
    if n.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)


def split_sentences(line, max_chars=400):
    out = []
    for s in re.split(r"(?<=[.!?।。！？؟])\s*", line):
        s = s.strip()
        if s:
            out += textwrap.wrap(s, max_chars, break_long_words=False) if len(s) > max_chars else [s]
    return out


def translate(text, src, tgt, batch_size, progress):
    tok, model, device = load_model()
    tok.src_lang = LANGS[src]
    bos = tok.convert_tokens_to_ids(LANGS[tgt])
    lines = text.splitlines()
    units = [(li, s) for li, ln in enumerate(lines) if ln.strip() for s in split_sentences(ln)]
    order = sorted(range(len(units)), key=lambda i: len(units[i][1]))  # length-sort = less padding
    outs = [""] * len(units)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([units[i][1] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        with torch.inference_mode():
            ids = model.generate(**enc, forced_bos_token_id=bos, num_beams=1,
                                 max_new_tokens=int(enc["input_ids"].shape[1] * 1.6) + 10)
        for i, t in zip(idx, tok.batch_decode(ids, skip_special_tokens=True)):
            outs[i] = t
        progress.progress(min(1.0, (start + batch_size) / max(1, len(order))))
    rebuilt = {}
    for (li, _), t in zip(units, outs):
        rebuilt.setdefault(li, []).append(t)
    return "\n".join(" ".join(rebuilt[i]) if i in rebuilt else "" for i in range(len(lines)))


def make_pdf(text, lang):
    font = "Helvetica"
    for f in FONTS.get(lang, []) + DEFAULT_FONTS:
        path = os.path.join(WIN, f)
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("UF", path))
                font = "UF"
                break
            except Exception:
                pass
    style = ParagraphStyle("u", parent=getSampleStyleSheet()["Normal"], fontName=font, fontSize=11, leading=16)
    rtl = lang == "Arabic"
    if rtl:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
        except ImportError:
            rtl = False
    out = io.BytesIO()
    story = []
    for line in text.splitlines():
        if rtl and line.strip():
            line = get_display(arabic_reshaper.reshape(line))
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or " "
        story += [Paragraph(safe, style), Spacer(1, 4)]
    SimpleDocTemplate(out).build(story)
    return out.getvalue()


def make_docx(text):
    out, doc = io.BytesIO(), Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(out)
    return out.getvalue()


# ---------------- UI ----------------
ss = st.session_state
ss.setdefault("history", {})   # thread_id -> record
ss.setdefault("current", None)

c1, c2 = st.columns(2)
source = c1.selectbox("Input language", list(LANGS), index=0)
target = c2.selectbox("Translate into", list(LANGS), index=1)
uploaded = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
prompt = st.text_area("Or enter text to translate", height=150)
batch_size = st.select_slider("Batch size (higher = faster, more RAM)", [4, 8, 16, 32], value=8)

if st.button("🚀 Translate", type="primary"):
    text = read_file(uploaded) if uploaded else prompt
    if not text.strip():
        st.error("Upload a file or enter text first.")
    elif source == target:
        st.warning("Choose different input and output languages.")
    else:
        try:
            with st.spinner("Loading NLLB-200 model (first run may take time)..."):
                load_model()
            bar = st.progress(0)
            result = translate(text, source, target, batch_size, bar)
            bar.empty()
            tid = str(uuid.uuid4())
            ss.history[tid] = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               "source": source, "target": target, "original": text, "translated": result}
            ss.current = tid
        except Exception as e:
            st.error(f"Translation failed: {e}")

# Output lives outside the button block so downloads don't wipe it
if ss.current:
    rec = ss.history[ss.current]
    st.success(f"Translation completed! Thread ID: {ss.current}")
    st.text_area("Final translation", rec["translated"], height=300)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ PDF", make_pdf(rec["translated"], rec["target"]), "translation.pdf", "application/pdf")
    d2.download_button("⬇️ DOCX", make_docx(rec["translated"]), "translation.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    d3.download_button("⬇️ TXT", rec["translated"], "translation.txt", "text/plain")

with st.expander("Translation history / import by thread ID"):
    tid_in = st.text_input("Thread ID")
    if st.button("Load thread") and tid_in.strip() in ss.history:
        ss.current = tid_in.strip()
        st.rerun()
    for t, r in reversed(list(ss.history.items())):
        st.write(f"`{t}` — {r['time']} — {r['source']} → {r['target']}")
    if not ss.history:
        st.write("No translations yet.")