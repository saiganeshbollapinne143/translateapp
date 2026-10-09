import io, uuid, time, os, json, re
from datetime import datetime
from xml.sax.saxutils import escape
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", page_icon="🌐", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#edf5ff,#f5f0ff,#e6fffa)}
.block-container{max-width:1200px;padding-top:1.5rem}
.hero{background:linear-gradient(120deg,#101e50,#2458b8,#087e8b);padding:26px;border-radius:18px;color:white;text-align:center;margin-bottom:20px}
.hero h1{font-size:clamp(25px,4vw,42px);font-weight:850;margin:0}
.hero p{color:#e0f7ff;margin:8px 0 0}
div.stButton>button{background:linear-gradient(90deg,#2155cc,#087e8b);color:white;border:0;border-radius:10px;font-weight:bold;min-height:45px}
div.stDownloadButton>button{border-radius:9px;border:1px solid #2458b8;color:#17418b}
</style><div class="hero"><h1>🌐 AIT GLOBAL TECHNOLOGIES</h1><p>AI-Powered Multilingual Document Translator</p></div>""", unsafe_allow_html=True)

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"English":"eng_Latn","Tamil":"tam_Taml","Spanish":"spa_Latn","French":"fra_Latn","German":"deu_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Gujarati":"guj_Gujr","Punjabi":"pan_Guru","Urdu":"urd_Arab","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Korean":"kor_Hang","Russian":"rus_Cyrl","Portuguese":"por_Latn","Italian":"ita_Latn","Dutch":"nld_Latn","Turkish":"tur_Latn","Thai":"tha_Thai","Vietnamese":"vie_Latn","Indonesian":"ind_Latn"}

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

if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history = []
if "translated" not in st.session_state: st.session_state.translated = ""

with st.sidebar:
    st.header("⚙️ Translation settings")
    source = st.selectbox("Translate from", list(LANGS), index=list(LANGS).index("English"), key="source")
    target = st.selectbox("Translate to", list(LANGS), index=list(LANGS).index("Tamil"), key="target")
    chunk_size = st.slider("Chunk size", 500, 2500, 1200, 100, key="chunk")
    max_tokens = st.slider("Maximum output tokens", 128, 512, 256, 32, key="tokens")
    st.subheader("🧩 Prompt & system")
    st.text_area("Prompt template (must contain {text})", value="{text}", key="template", height=90)
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
        st.rerun()

st.caption(f"**Thread ID:** `{st.session_state.thread_id}`")
left, right = st.columns(2)
with left:
    st.subheader("📝 Translation prompt")
    prompt = st.text_area("Enter English text or text in the selected source language", height=230, placeholder="Type or paste your text here...", key="input_text")
    uploaded = st.file_uploader("Or upload a TXT file", type=["txt"], key="input_txt")
    text_input = uploaded.getvalue().decode("utf-8-sig", errors="replace") if uploaded else prompt
    if uploaded: st.caption(f"Loaded {uploaded.name} · {len(text_input)} characters")
with right:
    st.subheader("🌍 Translation preview")
    if st.session_state.translated:
        st.text_area("Translated text", st.session_state.translated, height=230)
    else: st.info("Your translation will appear here.")

def split_chunks(text, size):
    chunks, current = [], ""
    for para in text.splitlines():
        para = para.strip()
        if not para: continue
        while len(para) > size:
            if current: chunks.append(current); current = ""
            chunks.append(para[:size]); para = para[size:]
        if current and len(current) + len(para) + 1 > size:
            chunks.append(current); current = para
        else: current = (current + "\n" + para).strip()
    if current: chunks.append(current)
    return chunks

def create_pdf(text, src, dst, thread_id, note):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    font_name = "Helvetica"
    font_path = "NotoSansTamil-Regular.ttf"
    if os.path.exists(font_path):
        try:
            if "NotoTamil" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("NotoTamil", font_path))
            font_name = "NotoTamil"
        except Exception:
            pass
    title_style = ParagraphStyle("Company", parent=styles["Title"], fontName=font_name, textColor=colors.HexColor("#2458B8"), fontSize=18, leading=24, spaceAfter=12)
    body_style = ParagraphStyle("TranslationBody", parent=styles["BodyText"], fontName=font_name, fontSize=11, leading=17, wordWrap="CJK")
    story = [Paragraph("AIT GLOBAL TECHNOLOGIES", title_style),
             Paragraph(escape(src + " to " + dst + " Translation"), body_style),
             Paragraph("Thread ID: " + escape(thread_id), body_style),
             Paragraph("Generated: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"), body_style)]
    if note.strip():
        story.append(Paragraph(escape(note), body_style))
    story.append(Spacer(1, 18))
    for para in text.splitlines():
        if para.strip():
            story.extend([Paragraph(escape(para), body_style), Spacer(1, 7)])
    doc.build(story)
    return buf.getvalue(), font_name == "NotoTamil"

if st.button("🚀 TRANSLATE & PREPARE PDF", type="primary", key="translate"):
    tpl = st.session_state.get("template", "{text}")
    if not text_input.strip(): st.warning("Enter text or upload a TXT file.")
    elif source == target: st.warning("Choose different source and target languages.")
    elif "{text}" not in tpl: st.warning("Prompt template must contain {text}.")
    else:
        try:
            started = time.time()
            progress = st.progress(0, text="Loading NLLB-200 model...")
            status = st.empty()
            tok, model = load_model()
            tok.src_lang = LANGS[source]
            final_text = tpl.replace("{text}", text_input)
            chunks = split_chunks(final_text, chunk_size)
            results = []
            for i, chunk in enumerate(chunks):
                inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
                with torch.inference_mode():
                    ids = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[target]), max_new_tokens=max_tokens, num_beams=1, do_sample=False)
                results.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
                progress.progress((i + 1) / len(chunks), text=f"Translating chunk {i+1}/{len(chunks)}")
                status.markdown("**Live translation:**\n\n" + "\n\n".join(results))
            st.session_state.translated = "\n\n".join(results)
            elapsed = round(time.time() - started, 2)
            st.session_state.history.insert(0, {"id":st.session_state.thread_id,"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"source":source,"target":target,"seconds":elapsed,"text":st.session_state.translated})
            save_thread()
            st.success(f"Translation completed in {elapsed} seconds.")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.translated:
    st.subheader("📄 Download translated PDF")
    try:
        pdf_bytes, tamil_font_ok = create_pdf(st.session_state.translated, source, target,
                                              st.session_state.thread_id,
                                              st.session_state.get("system_note", ""))
        st.download_button("⬇️ DOWNLOAD TRANSLATED PDF", data=pdf_bytes, file_name=f"AIT_Translated_{st.session_state.thread_id[:8]}.pdf", mime="application/pdf", key="download_pdf")
        if target == "Tamil" and not tamil_font_ok:
            st.warning("PDF created, but Tamil characters may not render correctly. Add NotoSansTamil-Regular.ttf next to app.py and redeploy.")
    except Exception as e: st.error(f"PDF generation failed: {e}")
    st.download_button("⬇️ Download TXT backup", st.session_state.translated.encode("utf-8"), file_name="translated.txt", mime="text/plain", key="download_txt")

st.divider()
st.subheader("📊 Session details")
a,b,c = st.columns(3)
a.metric("Translations", len(st.session_state.history))
b.metric("Input characters", len(text_input))
c.metric("Output characters", len(st.session_state.translated))
with st.expander("🕘 Translation history"):
    if not st.session_state.history: st.caption("No translations yet.")
    for item in st.session_state.history:
        st.markdown(f"**{item['source']} → {item['target']}** · {item['time']} · {item['seconds']} sec")
        st.caption("Thread ID: " + item["id"])
        st.text(item["text"][:1000])