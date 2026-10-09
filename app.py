
import io, os, re, json, time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AIT Global Translator", page_icon="🌐", layout="centered")
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#06152E,#102C54,#0A1F3D);color:#fff}
.main .block-container{padding-top:3.2rem;max-width:900px}
.title-card{background:linear-gradient(120deg,#082653,#164E91);border:1px solid #F4C542;border-radius:13px;padding:15px 8px;text-align:center;margin-bottom:22px;white-space:nowrap;overflow-x:auto}
.title-card h2{font-size:21px!important;color:#FFD447!important;margin:0;font-weight:800}
.title-card p{font-size:12px;color:white;margin:5px 0 0}
h1,h2,h3,label,p{color:#EAF2FF}
.stButton>button,.stDownloadButton>button{background:#F4C542;color:#082653;border:0;border-radius:8px;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{background:#fff;color:#082653}
div[data-testid="stFileUploader"]{background:#F0F5FC;color:#102C54;padding:10px;border-radius:10px}
</style>
<div class="title-card"><h2>AIT GLOBAL TECHNOLOGIES — TRANSLATOR</h2>
<p>AI-Powered Multilingual Document Translation</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    name="facebook/nllb-200-distilled-600M"
    token=os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN","")
    tok=AutoTokenizer.from_pretrained(name, token=token or None)
    model=AutoModelForSeq2SeqLM.from_pretrained(name, token=token or None)
    return tok,model

LANG={"English":"eng_Latn","Tamil":"tam_Taml","Hindi":"hin_Deva","Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym","French":"fra_Latn","Spanish":"spa_Latn","German":"deu_Latn","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Arabic":"arb_Arab","Portuguese":"por_Latn","Russian":"rus_Cyrl","Bengali":"ben_Beng"}
def extract(f):
    ext=f.name.lower()
    if ext.endswith(".txt"): return f.getvalue().decode("utf-8",errors="replace")
    if ext.endswith(".pdf"): return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    if ext.endswith(".docx"): return "\n".join(p.text for p in Document(f).paragraphs)
    if ext.endswith(".json"): return json.dumps(json.load(f),ensure_ascii=False,indent=2)
    raise ValueError("Upload TXT, PDF, DOCX or JSON.")
def translate(text,src,dst,tok,model,progress):
    tok.src_lang=LANG[src]; chunks=re.split(r"\n+",text)
    out=[]; chunks=[c.strip() for c in chunks if c.strip()]
    if not chunks: return ""
    for i,chunk in enumerate(chunks):
        words=chunk.split(); parts=[" ".join(words[j:j+180]) for j in range(0,len(words),180)]
        for part in parts:
            inputs=tok(part,return_tensors="pt",truncation=True,max_length=512)
            with torch.no_grad():
                ids=model.generate(**inputs,forced_bos_token_id=tok.convert_tokens_to_ids(LANG[dst]),max_new_tokens=256)
            out.append(tok.batch_decode(ids,skip_special_tokens=True)[0])
        progress.progress((i+1)/len(chunks),text=f"Translating section {i+1}/{len(chunks)}")
    return "\n".join(out)
def make_docx(text):
    b=io.BytesIO(); d=Document()
    for line in text.splitlines(): d.add_paragraph(line)
    d.save(b); return b.getvalue()
def make_pdf(text):
    b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4)
    styles=getSampleStyleSheet(); story=[]
    for line in text.splitlines():
        story.extend([Paragraph(line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") or " ",styles["Normal"]),Spacer(1,6)])
    doc.build(story); return b.getvalue()

uploaded=st.file_uploader("Upload source file",type=["txt","pdf","docx","json"])
c1,c2=st.columns(2)
src=c1.selectbox("Source language",list(LANG),index=0)
dst=c2.selectbox("Target language",list(LANG),index=1)
if st.button("🌐 Translate Document",use_container_width=True):
    if not uploaded: st.warning("Please upload a file first.")
    elif src==dst: st.warning("Choose different source and target languages.")
    else:
        try:
            with st.spinner("Loading translation model..."):
                tok,model=load_model()
            content=extract(uploaded)
            if not content.strip(): st.error("No readable text found in the file.")
            else:
                bar=st.progress(0,text="Starting translation...")
                result=translate(content,src,dst,tok,model,bar)
                st.session_state["translated"]=result
                st.success("Translation completed!")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.get("translated"):
    result=st.session_state["translated"]
    st.text_area("Translated output",result,height=250)
    st.download_button("⬇ Download TXT",result,"translation.txt","text/plain",use_container_width=True)
    st.download_button("⬇ Download PDF",make_pdf(result),"translation.pdf","application/pdf",use_container_width=True)
    st.download_button("⬇ Download DOCX",make_docx(result),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
    st.download_button("⬇ Download JSON",json.dumps({"translation":result},ensure_ascii=False,indent=2),"translation.json","application/json",use_container_width=True)
