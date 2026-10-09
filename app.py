
import io, uuid, re, html
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AI Translator", layout="wide")
st.title("🌍 NLLB-200 Document Translator")
MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"English":"eng_Latn","Tamil":"tam_Taml","Hindi":"hin_Deva","Telugu":"tel_Telu","Malayalam":"mal_Mlym","Kannada":"kan_Knda","French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Portuguese":"por_Latn","Russian":"rus_Cyrl","Bengali":"ben_Beng","Urdu":"urd_Arab"}

if "history" not in st.session_state: st.session_state.history = []
if "thread" not in st.session_state: st.session_state.thread = str(uuid.uuid4())
if "result" not in st.session_state: st.session_state.result = ""

@st.cache_resource(show_spinner="Loading NLLB model...")
def load_model():
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()
    return tok, model

def chunks(text, size=350):
    parts, current = [], ""
    for sentence in re.split(r'(?<=[.!?।])\s+', text.strip()):
        while len(sentence) > size:
            if current: parts.append(current); current = ""
            parts.append(sentence[:size]); sentence = sentence[size:]
        if len(current) + len(sentence) + 1 > size and current:
            parts.append(current); current = sentence
        else: current = (current + " " + sentence).strip()
    if current: parts.append(current)
    return parts

def pdf_bytes(text):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(html.escape(p), styles["Normal"]) for p in text.splitlines() if p.strip()]
    doc.build(story or [Paragraph("No translated text", styles["Normal"])])
    return out.getvalue()

c1, c2 = st.columns(2)
with c1: source = st.selectbox("Source language", list(LANGS))
with c2: target = st.selectbox("Target language", list(LANGS), index=1)
uploaded = st.file_uploader("Upload TXT file", type=["txt"])
prompt = st.text_area("Enter text to translate", height=120)
text = prompt.strip()
if uploaded and not text:
    text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
st.caption(f"Thread ID: {st.session_state.thread}")
if st.button("New thread"):
    st.session_state.thread = str(uuid.uuid4())
    st.rerun()

if st.button("Translate", type="primary"):
    if not text: st.warning("Enter text or upload a TXT file.")
    elif source == target: st.warning("Select different languages.")
    else:
        try:
            tok, model = load_model()
            tok.src_lang = LANGS[source]
            pieces = chunks(text)
            output, bar, status = [], st.progress(0), st.empty()
            for i, piece in enumerate(pieces):
                status.info(f"Translating chunk {i+1}/{len(pieces)}...")
                inputs = tok(piece, return_tensors="pt", truncation=True, max_length=512)
                with torch.inference_mode():
                    ids = model.generate(**inputs,
                        forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[target]),
                        max_new_tokens=256, num_beams=2, do_sample=False)
                output.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
                bar.progress((i+1)/len(pieces))
                status.info(f"Completed {i+1}/{len(pieces)} chunks")
                st.text_area("Live translation", "\n\n".join(output), height=150,
                             key=f"live_{st.session_state.thread}_{i}")
            result = "\n\n".join(output)
            st.session_state.result = result
            st.session_state.history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "thread": st.session_state.thread, "source": source, "target": target, "text": result})
            status.success("Translation completed!")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

if st.session_state.result:
    st.subheader("Translated document")
    st.text_area("Final output", st.session_state.result, height=220)
    st.download_button("Download TXT", st.session_state.result, "translated.txt", "text/plain")
    st.download_button("Download PDF", pdf_bytes(st.session_state.result), "translated.pdf", "application/pdf")

with st.expander("Translation history"):
    for item in reversed(st.session_state.history):
        st.write(f"{item['time']} | {item['source']} → {item['target']} | Thread: {item['thread']}")
        st.text(item["text"][:500])


