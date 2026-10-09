
import io, uuid, re, streamlit as st, torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES", layout="wide")
st.title("🌐 AIT GLOBAL TECHNOLOGIES — Translator")
MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"English":"eng_Latn","Spanish":"spa_Latn","French":"fra_Latn","German":"deu_Latn","Hindi":"hin_Deva","Tamil":"tam_Taml","Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Portuguese":"por_Latn"}
@st.cache_resource
def load_model():
    t=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForSeq2SeqLM.from_pretrained(MODEL); m.eval(); return t,m

c1,c2=st.columns(2)
src=c1.selectbox("Translate from",list(LANGS))
dst=c2.selectbox("Translate to",list(LANGS),index=1)
template=st.text_area("Manual prompt template","Translate the following text from {source_language} to {target_language}:\n\n{text}",height=90)
text=st.text_area("Paste input text",height=160)
file=st.file_uploader("Or upload TXT / PDF / DOCX",type=["txt","pdf","docx"])
if file:
    try:
        if file.name.lower().endswith(".txt"): text=file.getvalue().decode("utf-8-sig","replace")
        elif file.name.lower().endswith(".docx"): text="\n".join(p.text for p in Document(file).paragraphs)
        else:
            from pypdf import PdfReader
            text="\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
        st.info("Input document loaded.")
    except Exception as e: st.error(f"File error: {e}")
if "tid" not in st.session_state: st.session_state.tid=str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history=[]
st.caption("Thread ID: "+st.session_state.tid)
if st.button("🆕 New thread"): st.session_state.tid=str(uuid.uuid4()); st.rerun()

if st.button("🚀 Translate and prepare PDF + DOCX",type="primary",use_container_width=True):
    if not text.strip(): st.warning("Enter text or upload a document.")
    elif src==dst: st.warning("Choose different languages.")
    elif "{text}" not in template: st.warning("Template must contain {text}.")
    else:
        try:
            tok,model=load_model(); tok.src_lang=LANGS[src]
            chunks=re.split(r"(?<=[.!?।。])\s+",text.strip()); chunks=[x for x in chunks if x]
            out=[]; bar=st.progress(0); live=st.empty()
            for i,ch in enumerate(chunks):
                # NLLB uses text, not custom prompt instructions
                inputs=tok(ch,return_tensors="pt",truncation=True,max_length=512)
                with torch.inference_mode():
                    ids=model.generate(**inputs,forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[dst]),max_new_tokens=512,num_beams=1)
                out.append(tok.batch_decode(ids,skip_special_tokens=True)[0])
                live.text_area("Streaming translation", "\n\n".join(out),height=230)
                bar.progress((i+1)/len(chunks),text=f"Translating chunk {i+1}/{len(chunks)}")
            st.session_state.output="\n\n".join(out)
            st.session_state.history.insert(0,{"thread":st.session_state.tid,"time":datetime.now().isoformat(timespec="seconds"),"from":src,"to":dst})
            st.success("Translation completed.")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.get("output"):
    result=st.session_state.output
    pdf=io.BytesIO(); styles=getSampleStyleSheet()
    story=[Paragraph("AIT GLOBAL TECHNOLOGIES",styles["Title"]),Paragraph(f"{src} → {dst} | Thread: {st.session_state.tid}",styles["Normal"]),Spacer(1,12)]
    story += [Paragraph(escape(p),styles["BodyText"]) for p in result.splitlines() if p.strip()]
    SimpleDocTemplate(pdf,pagesize=A4).build(story)
    doc=Document(); doc.add_heading("AIT GLOBAL TECHNOLOGIES",0); doc.add_paragraph(f"{src} → {dst} | Thread: {st.session_state.tid}")
    for p in result.splitlines():
        if p.strip(): doc.add_paragraph(p)
    word=io.BytesIO(); doc.save(word)
    a,b=st.columns(2)
    a.download_button("⬇ Download PDF",pdf.getvalue(),"translation.pdf","application/pdf",use_container_width=True)
    b.download_button("⬇ Download DOCX",word.getvalue(),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
with st.expander("🕘 Translation history"): st.write(st.session_state.history)




