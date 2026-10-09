
import io, os, re, json
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AIT Global Translator", page_icon="🌐", layout="centered")
st.markdown("""<style>
.stApp{background:#fff;color:#183153}
.main .block-container{padding-top:3rem;max-width:900px}
.title-card{background:linear-gradient(120deg,#082653,#164E91);border:1px solid #F4C542;border-radius:13px;padding:15px 6px;text-align:center;margin-bottom:22px;white-space:nowrap;overflow-x:auto}
.title-card h2{font-size:19px!important;color:#FFD447!important;margin:0;font-weight:800}
.title-card p{font-size:12px;color:#fff;margin:5px 0 0}
h1,h2,h3,p,label{color:#183153}
.stButton>button,.stDownloadButton>button{background:#F4C542;color:#082653;border:1px solid #D9AC25;border-radius:8px;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{background:#164E91;color:white}
div[data-testid="stFileUploader"]{background:#F3F7FC;color:#183153;padding:10px;border-radius:10px;border:1px solid #D8E3F0}
</style><div class="title-card"><h2>AIT GLOBAL TECHNOLOGIES — TRANSLATOR</h2><p>AI-Powered Multilingual Document Translation</p></div>""",unsafe_allow_html=True)

LANG={"English":"eng_Latn","Tamil":"tam_Taml","Hindi":"hin_Deva","Telugu":"tel_Telu","Kannada":"kan_Knda","Malayalam":"mal_Mlym","French":"fra_Latn","Spanish":"spa_Latn","German":"deu_Latn","Chinese":"zho_Hans","Japanese":"jpn_Jpan","Arabic":"arb_Arab","Portuguese":"por_Latn","Russian":"rus_Cyrl","Bengali":"ben_Beng"}
@st.cache_resource
def load_model():
    name="facebook/nllb-200-distilled-600M"
    token=os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN","")
    tok=AutoTokenizer.from_pretrained(name,token=token or None)
    model=AutoModelForSeq2SeqLM.from_pretrained(name,token=token or None)
    return tok,model

def extract(f):
    n=f.name.lower()
    if n.endswith(".txt"): return f.getvalue().decode("utf-8",errors="replace")
    if n.endswith(".pdf"): return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    if n.endswith(".docx"): return "\n".join(p.text for p in Document(f).paragraphs)
    if n.endswith(".json"): return json.dumps(json.load(f),ensure_ascii=False,indent=2)
    raise ValueError("Unsupported file format")

def translate(text,src,dst,tok,model,bar):
    tok.src_lang=LANG[src]
    chunks=[x.strip() for x in re.split(r"\n+",text) if x.strip()]
    result=[]
    if not chunks: return ""
    for i,chunk in enumerate(chunks):
        words=chunk.split()
        for j in range(0,len(words),180):
            part=" ".join(words[j:j+180])
            inputs=tok(part,return_tensors="pt",truncation=True,max_length=512)
            with torch.no_grad():
                ids=model.generate(**inputs,forced_bos_token_id=tok.convert_tokens_to_ids(LANG[dst]),max_new_tokens=256)
            result.append(tok.batch_decode(ids,skip_special_tokens=True)[0])
        bar.progress((i+1)/len(chunks),text=f"Translating section {i+1}/{len(chunks)}")
    return "\n".join(result)

def make_docx(text):
    b=io.BytesIO(); d=Document()
    for line in text.splitlines(): d.add_paragraph(line)
    d.save(b); return b.getvalue()

def make_pdf(text):
    b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4)
    styles=getSampleStyleSheet()
    story=[Paragraph(line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") or " ",styles["Normal"]) for line in text.splitlines()]
    doc.build(story); return b.getvalue()

uploaded=st.file_uploader("Upload document",type=["pdf","txt","docx","json"])
a,b=st.columns(2)
src=a.selectbox("Source language",list(LANG))
dst=b.selectbox("Target language",list(LANG),index=1)

if st.button("🌐 Translate Document",use_container_width=True):
    if not uploaded: st.warning("Please upload a file.")
    elif src==dst: st.warning("Select different languages.")
    else:
        try:
            text=extract(uploaded)
            if not text.strip(): st.error("No readable text found.")
            else:
                with st.spinner("Loading AI translation model..."): tok,model=load_model()
                bar=st.progress(0,text="Starting translation...")
                result=translate(text,src,dst,tok,model,bar)
                if result:
                    st.session_state["translated"]=result
                    st.success("Translation completed!")
                else: st.warning("No text was available to translate.")
        except Exception as e: st.error(f"Translation failed: {e}")

if st.session_state.get("translated"):
    result=st.session_state["translated"]
    st.text_area("Translated output",result,height=220)
    st.download_button("⬇ Download TXT",result,"translation.txt","text/plain",use_container_width=True)
    st.download_button("⬇ Download PDF",make_pdf(result),"translation.pdf","application/pdf",use_container_width=True)
    st.download_button("⬇ Download DOCX",make_docx(result),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
    st.download_button("⬇ Download JSON",json.dumps({"translation":result},ensure_ascii=False,indent=2),"translation.json","application/json",use_container_width=True)
