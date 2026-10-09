import io, uuid, time, os, json, re
from datetime import datetime
from xml.sax.saxutils import escape
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph as DocxParagraph
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", page_icon="🌐", layout="wide")

# ---------- Light UI styling ----------
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#f4f8ff 0%,#f8f5ff 50%,#effcf9 100%)}
.block-container{max-width:1200px;padding-top:1.2rem}
section[data-testid="stSidebar"]{background:#ffffffcc;border-right:1px solid #e3e9f5}
.hero{background:linear-gradient(120deg,#e8f0ff,#efe9ff,#e2faf5);border:1px solid #dbe5f7;
      padding:24px;border-radius:18px;text-align:center;margin-bottom:18px;
      box-shadow:0 4px 18px rgba(36,88,184,.08)}
.hero h1{font-size:clamp(24px,4vw,38px);font-weight:800;margin:0;color:#1d3b8b}
.hero p{color:#4a5a7a;margin:6px 0 0}
.badges{margin-top:10px}
.badge{display:inline-block;background:#fff;border:1px solid #d5e0f5;color:#2458b8;
       padding:3px 12px;border-radius:999px;font-size:12px;font-weight:600;margin:0 3px}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#ffffffb8;border-radius:14px}
div[data-testid="stFileUploader"] section{background:#f7faff;border:1.5px dashed #9db8e8;border-radius:12px}
div.stButton>button{background:linear-gradient(90deg,#4f7de0,#2fa8b5);color:white;border:0;
                    border-radius:10px;font-weight:600;min-height:44px}
div.stButton>button:hover{filter:brightness(1.06);color:white}
div.stDownloadButton>button{border-radius:10px;border:1px solid #9db8e8;color:#1d3b8b;background:#fff}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e3e9f5;border-radius:12px;padding:10px 14px}
</style>
<div class="hero"><h1>🌐 AIT GLOBAL TECHNOLOGIES</h1>
<p>AI-Powered Multilingual Document Translator</p>
<div class="badges"><span class="badge">📄 PDF</span><span class="badge">📝 DOCX</span>
<span class="badge">📃 TXT</span><span class="badge">➜ PDF output</span></div></div>""",
            unsafe_allow_html=True)

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"English":"eng_Latn","Tamil":"tam_Taml","Spanish":"spa_Latn","French":"fra_Latn","German":"deu_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Gujarati":"guj_Gujr","Punjabi":"pan_Guru","Urdu":"urd_Arab","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Korean":"kor_Hang","Russian":"rus_Cyrl","Portuguese":"por_Latn","Italian":"ita_Latn","Dutch":"nld_Latn","Turkish":"tur_Latn","Thai":"tha_Thai","Vietnamese":"vie_Latn","Indonesian":"ind_Latn"}

# Put these .ttf files next to app.py to render non-Latin scripts in the PDF
FONT_FILES = {"Tamil": "NotoSansTamil-Regular.ttf",
              "Hindi": "NotoSansDevanagari-Regular.ttf",
              "Marathi": "NotoSansDevanagari-Regular.ttf",
              "Telugu": "NotoSansTelugu-Regular.ttf",
              "Kannada": "NotoSansKannada-Regular.ttf",
              "Malayalam": "NotoSansMalayalam-Regular.ttf"}

THREADS_DIR = "threads"
os.makedirs(THREADS_DIR, exist_ok=True)
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

@st.cache_resource
def load_model():
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()
    return tok, model

# ---------- File readers ----------
def read_txt(data):
    return data.decode("utf-8-sig", errors="replace")

def read_pdf(data):
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)      # join hard-wrapped lines
        t = re.sub(r"[ \t]+", " ", t).strip()
        if t: pages.append(t)
    return "\n".join(pages)

def read_docx(data):
    doc = Document(io.BytesIO(data))
    out = []
    for child in doc.element.body.iterchildren():   # keeps paragraphs/tables in order
        if child.tag.endswith("}p"):
            t = DocxParagraph(child, doc).text.strip()
            if t: out.append(t)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                last = None
                for cell in row.cells:
                    t = cell.text.strip()
                    if t and t != last: out.append(t)   # skip merged-cell repeats
                    last = t
    return "\n".join(out)

READERS = {"txt": read_txt, "pdf": read_pdf, "docx": read_docx}

def read_uploads(files):
    parts, names = [], []
    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        try:
            text = READERS[ext](f.getvalue())
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")
            continue
        if not text.strip():
            st.warning(f"{f.name}: no text found (a scanned PDF needs OCR first).")
            continue
        parts.append(text)
        names.append(f.name)
        st.caption(f"✅ {f.name} · {len(text):,} characters")
    return "\n".join(parts), names

# ---------- Threads ----------
def save_thread():
    tid = st.session_state.thread_id
    data = {"thread_id": tid,
            "history": st.session_state.history,
            "translated": st.session_state.translated,
            "template": st.session_state.get("template", "{text}"),
            "system_note": st.session_state.get("system_note", "")}
    with open(os.path.join(THREADS_DIR, f"{tid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def import_thread():
    tid = st.session_state.get("import_id", "").strip()
    path = os.path.join(THREADS_DIR, f"{tid}.json")
    if not UUID_RE.match(tid) or not os.path.exists(path):
        st.session_state.import_msg = ("error", "Thread ID not found.")
        return
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    st.session_state.thread_id = d["thread_id"]
    st.session_state.history = d.get("history", [])
    st.session_state.translated = d.get("translated", "")
    st.session_state.template = d.get("template", "{text}")
    st.session_state.system_note = d.get("system_note", "")
    st.session_state.import_msg = ("success", "Thread imported.")

PRESETS = {
    "Plain (no wrapper)": "{text}",
    "Formal letter": "Dear Sir/Madam,\n\n{text}\n\nYours faithfully,\nAIT Global Technologies",
    "Email": "Hello,\n\n{text}\n\nBest regards,\nAIT Global Technologies",
    "Official notice": "NOTICE\n\n{text}\n\nBy order of AIT Global Technologies",
    "Custom": None,
}

def apply_preset():
    tpl = PRESETS[st.session_state.preset]
    if tpl is not None:
        st.session_state.template = tpl

def llm_translate(prompt, api_key, model_name):
    """Send the full prompt (instructions + chunk) to Groq and return the reply."""
    for attempt in range(3):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}"},
                          json={"model": model_name, "temperature": 0.2, "max_tokens": 4096,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=120)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1)); continue
        if not r.ok:
            raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Groq rate limit reached. Wait a moment and try again.")

def load_prompt_file():
    f = st.session_state.get("prompt_file")
    if f is None: return
    ext = f.name.rsplit(".", 1)[-1].lower()
    try:
        text = READERS[ext](f.getvalue()).strip()
    except Exception as e:
        st.session_state.prompt_msg = ("error", f"Could not read {f.name}: {e}")
        return
    if not text:
        st.session_state.prompt_msg = ("error", f"{f.name} has no text.")
        return
    if "{text}" in text:
        st.session_state.prompt_msg = ("success", f"Prompt loaded from {f.name}.")
    else:
        text += "\n\n{text}"
        st.session_state.prompt_msg = ("warning", f"No {{text}} found in {f.name}, so it was added at the end.")
    st.session_state.template = text
    st.session_state.preset = "Custom"

if "template" not in st.session_state: st.session_state.template = "{text}"
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history = []
if "translated" not in st.session_state: st.session_state.translated = ""
if "sources" not in st.session_state: st.session_state.sources = []

with st.sidebar:
    st.header("⚙️ Translation settings")
    source = st.selectbox("Translate from", list(LANGS), index=list(LANGS).index("English"), key="source")
    target = st.selectbox("Translate to", list(LANGS), index=list(LANGS).index("Tamil"), key="target")
    engine = st.radio("Translation engine", ["NLLB (local)", "LLM (Groq) - follows your prompt"], key="engine")
    use_llm = engine.startswith("LLM")
    if use_llm:
        api_key = st.text_input("Groq API key", type="password", value=os.environ.get("GROQ_API_KEY", ""), key="groq_key")
        llm_model = st.text_input("Groq model", value="llama-3.3-70b-versatile", key="groq_model")
        st.caption("Tip: set Chunk size to about 2500 for the LLM.")
    else:
        api_key, llm_model = "", ""
    chunk_size = st.slider("Chunk size", 300, 2500, 800, 100, key="chunk")
    max_tokens = st.slider("Maximum output tokens", 128, 512, 256, 32, key="tokens")
    st.subheader("🧩 Prompt & system")
    st.selectbox("Template preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.file_uploader("Or upload a prompt file", type=["txt", "docx", "pdf"],
                     key="prompt_file", on_change=load_prompt_file)
    if "prompt_msg" in st.session_state:
        kind, msg = st.session_state.pop("prompt_msg")
        getattr(st, kind)(msg)
    st.text_area("Prompt template (must contain {text})", key="template", height=130)
    st.text_area("System note (added to PDF, not translated)", key="system_note", height=70,
                 placeholder="e.g. Official translation for AIT Global internal use")
    st.subheader("📥 Import thread")
    st.text_input("Thread ID", key="import_id")
    st.button("Import", key="import_btn", on_click=import_thread)
    if "import_msg" in st.session_state:
        kind, msg = st.session_state.pop("import_msg")
        getattr(st, kind)(msg)
    if st.button("🆕 New session", key="new_session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.translated = ""
        st.session_state.history = []
        st.session_state.sources = []
        st.rerun()

st.caption(f"**Thread ID:** `{st.session_state.thread_id}`")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("📝 Input")
        uploaded = st.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "docx", "txt"],
                                    accept_multiple_files=True, key="input_files")
        file_text, file_names = read_uploads(uploaded) if uploaded else ("", [])
        prompt = st.text_area("Or type / paste text", height=180,
                              placeholder="Type or paste your text here...", key="input_text",
                              disabled=bool(file_text))
        if file_text:
            st.info("Using uploaded file text. Remove the files to type manually.")
            with st.expander("👁️ Preview extracted text"):
                st.text(file_text[:3000] + ("..." if len(file_text) > 3000 else ""))
        text_input = file_text if file_text else prompt
with right:
    with st.container(border=True):
        st.subheader("🌍 Translation preview")
        if st.session_state.translated:
            st.text_area("Translated text", st.session_state.translated, height=300)
        else:
            st.info("Your translation will appear here.")

def split_chunks(text, size):
    """Paragraph-aware, sentence-based chunking (better quality for NLLB)."""
    chunks = []
    for para in text.splitlines():
        para = para.strip()
        if not para: continue
        current = ""
        for sent in re.split(r"(?<=[.!?।。])\s+", para):
            while len(sent) > size:
                if current: chunks.append(current); current = ""
                chunks.append(sent[:size]); sent = sent[size:]
            if current and len(current) + len(sent) + 1 > size:
                chunks.append(current); current = sent
            else:
                current = (current + " " + sent).strip()
        if current: chunks.append(current)
    return chunks

def get_font(target_lang):
    path = FONT_FILES.get(target_lang)
    if path and os.path.exists(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, path))
            return name, True
        except Exception:
            pass
    return "Helvetica", target_lang not in FONT_FILES

def create_pdf(text, src, dst, thread_id, note, sources):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48,
                            title="AIT Translation", author="AIT GLOBAL TECHNOLOGIES")
    styles = getSampleStyleSheet()
    font_name, font_ok = get_font(dst)
    blue = colors.HexColor("#2458B8")
    title_style = ParagraphStyle("Company", parent=styles["Title"], fontName="Helvetica-Bold",
                                 textColor=blue, fontSize=20, leading=26, spaceAfter=4)
    meta_style = ParagraphStyle("Meta", parent=styles["BodyText"], fontName="Helvetica",
                                fontSize=9, leading=13, textColor=colors.HexColor("#5a6b8c"))
    note_style = ParagraphStyle("Note", parent=meta_style, fontName=font_name, backColor=colors.HexColor("#eef4ff"),
                                borderPadding=6, spaceBefore=6, textColor=colors.HexColor("#1d3b8b"))
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontName=font_name,
                                fontSize=11, leading=17, wordWrap="CJK")
    story = [Paragraph("AIT GLOBAL TECHNOLOGIES", title_style),
             Paragraph(escape(f"{src} to {dst} Translation"), meta_style)]
    if sources:
        story.append(Paragraph("Source: " + escape(", ".join(sources)), meta_style))
    story += [Paragraph("Thread ID: " + escape(thread_id), meta_style),
              Paragraph("Generated: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"), meta_style)]
    if note.strip():
        story.append(Spacer(1, 6)); story.append(Paragraph(escape(note), note_style))
    story += [Spacer(1, 10), HRFlowable(width="100%", thickness=1, color=blue), Spacer(1, 12)]
    for para in text.splitlines():
        if para.strip():
            story.extend([Paragraph(escape(para), body_style), Spacer(1, 7)])
    doc.build(story)
    return buf.getvalue(), font_ok

if st.button("🚀 TRANSLATE & PREPARE PDF", type="primary", key="translate", use_container_width=True):
    tpl = st.session_state.get("template", "{text}")
    if not text_input.strip(): st.warning("Upload a PDF/DOCX/TXT file or enter some text.")
    elif source == target: st.warning("Choose different source and target languages.")
    elif "{text}" not in tpl: st.warning("Prompt template must contain {text}.")
    elif use_llm and not api_key.strip(): st.warning("Enter your Groq API key in the sidebar.")
    else:
        try:
            started = time.time()
            progress = st.progress(0, text="Loading NLLB-200 model...")
            status = st.empty()
            if use_llm:
                chunks = split_chunks(text_input, chunk_size)   # prompt is applied to each chunk
            else:
                tok, model = load_model()
                tok.src_lang = LANGS[source]
                chunks = split_chunks(tpl.replace("{text}", text_input), chunk_size)
            results = []
            for i, chunk in enumerate(chunks):
                if use_llm:
                    results.append(llm_translate(tpl.replace("{text}", chunk), api_key.strip(), llm_model.strip()))
                else:
                    inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
                    with torch.inference_mode():
                        ids = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[target]),
                                             max_new_tokens=max_tokens, num_beams=1, do_sample=False)
                    results.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
                progress.progress((i + 1) / len(chunks), text=f"Translating chunk {i+1}/{len(chunks)}")
                status.markdown("**Live translation:**\n\n" + "\n\n".join(results))
            st.session_state.translated = "\n\n".join(results)
            st.session_state.sources = file_names
            elapsed = round(time.time() - started, 2)
            st.session_state.history.insert(0, {"id": st.session_state.thread_id,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": source, "target": target,
                "seconds": elapsed, "text": st.session_state.translated})
            save_thread()
            st.success(f"Translation completed in {elapsed} seconds.")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.translated:
    with st.container(border=True):
        st.subheader("📄 Download translated PDF")
        try:
            pdf_bytes, font_ok = create_pdf(st.session_state.translated, source, target,
                                            st.session_state.thread_id,
                                            st.session_state.get("system_note", ""),
                                            st.session_state.sources)
            base = os.path.splitext(st.session_state.sources[0])[0] if st.session_state.sources else "translation"
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ DOWNLOAD TRANSLATED PDF", data=pdf_bytes,
                               file_name=f"{base}_{target}.pdf", mime="application/pdf",
                               key="download_pdf", use_container_width=True)
            d2.download_button("⬇️ Download TXT backup", st.session_state.translated.encode("utf-8"),
                               file_name=f"{base}_{target}.txt", mime="text/plain",
                               key="download_txt", use_container_width=True)
            if not font_ok:
                st.warning(f"PDF created, but {target} characters may not render correctly. "
                           f"Add {FONT_FILES.get(target, 'a suitable .ttf font')} next to app.py.")
        except Exception as e: st.error(f"PDF generation failed: {e}")

st.divider()
st.subheader("📊 Session details")
a, b, c = st.columns(3)
a.metric("Translations", len(st.session_state.history))
b.metric("Input characters", len(text_input))
c.metric("Output characters", len(st.session_state.translated))
with st.expander("🕘 Translation history"):
    if not st.session_state.history: st.caption("No translations yet.")
    for item in st.session_state.history:
        st.markdown(f"**{item['source']} → {item['target']}** · {item['time']} · {item['seconds']} sec")
        st.caption("Thread ID: " + item["id"])
        st.text(item["text"][:1000])