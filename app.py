
import io, uuid, time
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", page_icon="🌐", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#edf5ff,#f5f0ff,#e6fffa);}
.block-container {padding-top:1.5rem;max-width:1200px;}
.hero {background:linear-gradient(120deg,#101e50,#2458b8,#087e8b);
padding:30px;border-radius:20px;color:white;text-align:center;
box-shadow:0 8px 24px #163b6340;margin-bottom:22px;}
.hero h1 {font-size:clamp(25px,4vw,42px);font-weight:850;margin:0;}
.hero p {font-size:17px;color:#e0f7ff;margin:8px 0 0;}
div.stButton>button {background:linear-gradient(90deg,#2155cc,#087e8b);
color:white;border:0;border-radius:10px;font-weight:bold;min-height:45px;}
div.stDownloadButton>button {border-radius:9px;border:1px solid #2458b8;color:#17418b;}
[data-testid="stMetric"] {background:white;padding:12px;border-radius:12px;
box-shadow:0 2px 10px #18365b12;}
</style>
<div class="hero">
<h1>🌐 AIT GLOBAL TECHNOLOGIES</h1>
<p>AI-Powered Multilingual Document Translator</p>
</div>
""", unsafe_allow_html=True)

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {
    "English":"eng_Latn","Spanish":"spa_Latn","French":"fra_Latn",
    "German":"deu_Latn","Hindi":"hin_Deva","Tamil":"tam_Taml",
    "Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym",
    "Bengali":"ben_Beng","Marathi":"mar_Deva","Gujarati":"guj_Gujr",
    "Punjabi":"pan_Guru","Urdu":"urd_Arab","Arabic":"arb_Arab",
    "Chinese (Simplified)":"zho_Hans","Japanese":"jpn_Jpan",
    "Korean":"kor_Hang","Russian":"rus_Cyrl","Portuguese":"por_Latn",
    "Italian":"ita_Latn","Dutch":"nld_Latn","Turkish":"tur_Latn",
    "Thai":"tha_Thai","Vietnamese":"vie_Latn","Indonesian":"ind_Latn"
}

@st.cache_resource
def load_model():
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()
    return tokenizer, model

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "translated" not in st.session_state:
    st.session_state.translated = ""

with st.sidebar:
    st.header("⚙️ Translation Settings")
    source = st.selectbox("Translate from", list(LANGS), key="source")
    target = st.selectbox("Translate to", list(LANGS), index=1, key="target")
    chunk_size = st.slider("Chunk size (characters)", 500, 2500, 1200, 100, key="chunk")
    max_tokens = st.slider("Max output tokens per chunk", 128, 512, 256, 32, key="tokens")
    st.caption("CPU optimized: greedy decoding and cached model.")
    if st.button("🆕 New translation session", key="new_session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.translated = ""
        st.rerun()

st.caption(f"**Thread ID:** `{st.session_state.thread_id}`")
left, right = st.columns(2)
with left:
    st.subheader("📝 Input document")
    text_input = st.text_area("Type or paste text", height=230, key="input_text",
                              placeholder="Enter text to translate...")
    uploaded = st.file_uploader("Or upload a TXT file", type=["txt"], key="input_txt")
    if uploaded:
        text_input = uploaded.getvalue().decode("utf-8-sig", errors="replace")
        st.caption(f"Loaded: {uploaded.name} · {len(text_input)} characters")
with right:
    st.subheader("🌍 Translation output")
    output_area = st.empty()
    if st.session_state.translated:
        output_area.text_area("Translated text", st.session_state.translated,
                              height=230, key="output_preview")
    else:
        output_area.info("Your translated text will appear here.")

def make_chunks(text, size):
    # Keep paragraphs together where possible; split oversized paragraphs.
    chunks, current = [], ""
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size])
            paragraph = paragraph[size:]
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = paragraph
        else:
            current = (current + "\n" + paragraph).strip()
    if current:
        chunks.append(current)
    return chunks

if st.button("🚀 TRANSLATE DOCUMENT", type="primary", key="translate"):
    if not text_input.strip():
        st.warning("Enter text or upload a TXT file.")
    elif source == target:
        st.warning("Select different source and target languages.")
    else:
        try:
            start = time.time()
            progress = st.progress(0, text="Loading translation model...")
            status = st.empty()
            tokenizer, model = load_model()
            tokenizer.src_lang = LANGS[source]
            chunks = make_chunks(text_input, chunk_size)
            results = []
            for i, chunk in enumerate(chunks):
                inputs = tokenizer(chunk, return_tensors="pt",
                                   truncation=True, max_length=512)
                with torch.inference_mode():
                    ids = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANGS[target]),
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=False
                    )
                results.append(tokenizer.batch_decode(ids, skip_special_tokens=True)[0])
                progress.progress((i + 1) / len(chunks),
                                  text=f"Translating chunk {i+1}/{len(chunks)}")
                status.markdown("**Live output:**\n\n" + "\n\n".join(results))
            final_text = "\n\n".join(results)
            elapsed = round(time.time() - start, 2)
            st.session_state.translated = final_text
            st.session_state.history.insert(0, {
                "id": st.session_state.thread_id,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source, "target": target,
                "seconds": elapsed, "text": final_text
            })
            st.success(f"Translation completed in {elapsed} seconds.")
            st.rerun()
        except Exception as exc:
            st.error(f"Translation error: {exc}")

if st.session_state.translated:
    result = st.session_state.translated
    st.subheader("📥 Download translated document")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download TXT", result.encode("utf-8"),
                           file_name="translated.txt", mime="text/plain",
                           key="download_txt")
    with c2:
        pdf_buffer = io.BytesIO()
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.styles import ParagraphStyle
            from xml.sax.saxutils import escape
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                    rightMargin=40, leftMargin=40,
                                    topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            story = [Paragraph("AIT GLOBAL TECHNOLOGIES", styles["Title"]),
                     Spacer(1, 16)]
            for para in result.splitlines():
                if para.strip():
                    story.append(Paragraph(escape(para), styles["BodyText"]))
                    story.append(Spacer(1, 7))
            doc.build(story)
            st.download_button("⬇️ Download PDF", pdf_buffer.getvalue(),
                               file_name="translated.pdf", mime="application/pdf",
                               key="download_pdf")
        except Exception as exc:
            st.warning(f"PDF generation failed: {exc}. Download TXT instead.")

st.divider()
st.subheader("📊 Session details")
m1, m2, m3 = st.columns(3)
m1.metric("Translations this session", len(st.session_state.history))
m2.metric("Input characters", len(text_input))
m3.metric("Output characters", len(st.session_state.translated))
with st.expander("🕘 Translation history"):
    if not st.session_state.history:
        st.caption("No translations yet.")
    for item in st.session_state.history:
        st.markdown(f"**{item['source']} → {item['target']}** · {item['time']} · {item['seconds']} sec")
        st.caption(f"Thread ID: {item['id']}")
        st.text(item["text"][:1000])
