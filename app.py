
import io, re, json, uuid, html
from datetime import datetime
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

st.set_page_config(page_title="AIT GLOBAL TECHNOLOGIES TRANSLATOR",
                   page_icon="🌐", layout="wide")

st.markdown("""
<style>
.stApp{background:#F3F7FC;color:#243247}
.block-container{max-width:1150px;padding-top:1.5rem}
.title-card{background:linear-gradient(135deg,#123B70,#287ED0);
color:white;text-align:center;padding:25px 15px;border-radius:18px;
margin-bottom:25px;box-shadow:0 8px 24px #174A8325}
.brand{color:#DCEBFF;font-size:12px;font-weight:800;letter-spacing:4px}
.title-card h1{color:white;font-size:clamp(23px,4vw,35px);
font-weight:850;margin:10px 0}
.subtitle{color:#E5F0FF;font-size:13px}
h2,h3{color:#174A83!important}
.stApp label{color:#263B53!important;font-weight:600}
div[data-testid="stFileUploader"]{background:white;border:2px dashed #8BB9EA;
border-radius:13px;padding:12px}
div[data-testid="stTextArea"] textarea{background:white;color:#243247;
border:1px solid #CBD8E8;border-radius:10px}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{
background:white;border-radius:9px}
.stButton>button,.stDownloadButton>button{min-height:43px;
border-radius:10px;font-weight:700;transition:all .2s}
.stButton>button{background:linear-gradient(135deg,#174A83,#287ED0);
color:white;border:1px solid #174A83}
.stButton>button:hover{background:#0F315A;color:white;
transform:translateY(-1px)}
.stDownloadButton>button{background:white;color:#17569A;
border:1px solid #8BB9EA}
.stDownloadButton>button:hover{background:#E6F1FF;color:#123B70;
border-color:#2476C7}
div[data-testid="stProgress"]>div>div>div{
background:linear-gradient(90deg,#2476C7,#52A5F5)}
div[data-testid="stExpander"]{background:white;border:1px solid #D7E3F1;
border-radius:10px}
</style>
<div class="title-card">
<div class="brand">AIT GLOBAL TECHNOLOGIES</div>
<h1>🌐 AIT GLOBAL TECHNOLOGIES<br>TRANSLATOR</h1>
<div class="subtitle">Smart Document Translation · PDF · DOCX · TXT · JSON</div>
</div>
""", unsafe_allow_html=True)

MODEL="facebook/nllb-200-distilled-600M"
LANGS={"English":"eng_Latn","Tamil":"tam_Taml","Hindi":"hin_Deva",
"Telugu":"tel_Telu","Malayalam":"mal_Mlym","Kannada":"kan_Knda",
"French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn",
"Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan",
"Portuguese":"por_Latn","Russian":"rus_Cyrl","Bengali":"ben_Beng",
"Urdu":"urd_Arab"}

if "history" not in st.session_state: st.session_state.history=[]
if "thread" not in st.session_state: st.session_state.thread=str(uuid.uuid4())
if "result" not in st.session_state: st.session_state.result=None

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()
    return tok,model

def read_file(f):
    name=f.name.lower()
    if name.endswith(".pdf"):
        text="\n\n".join(p.extract_text() or "" for p in
                         PdfReader(io.BytesIO(f.getvalue())).pages)
        if not text.strip(): raise ValueError("Scanned PDF needs OCR.")
        return "text",text
    if name.endswith(".docx"):
        d=Document(io.BytesIO(f.getvalue()))
        lines=[p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
        return "text","\n".join(lines)
    raw=f.getvalue().decode("utf-8-sig")
    return ("json",json.loads(raw)) if name.endswith(".json") else ("text",raw)

def chunks(text,size=300):
    parts=[]; current=""
    for s in re.split(r"(?<=[.!?।])\s+",text.strip()):
        while len(s)>size:
            if current: parts.append(current); current=""
            parts.append(s[:size]); s=s[size:]
        if current and len(current)+len(s)+1>size:
            parts.append(current); current=s
        else: current=(current+" "+s).strip()
    if current: parts.append(current)
    return parts

def pdf_bytes(text):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,
        rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    styles=getSampleStyleSheet(); story=[]
    for line in text.splitlines():
        if line.strip():
            story += [Paragraph(html.escape(line),styles["Normal"]),Spacer(1,6)]
    doc.build(story or [Paragraph("No translated text",styles["Normal"])])
    return buf.getvalue()

def docx_bytes(text):
    d=Document()
    for line in text.splitlines(): d.add_paragraph(line)
    buf=io.BytesIO(); d.save(buf); return buf.getvalue()

def translate_string(text,tok,model,target,progress,status,live,counter,total):
    output=[]
    for piece in chunks(text):
        counter[0]+=1
        status.info(f"Translating chunk {counter[0]} of {total}...")
        inputs=tok(piece,return_tensors="pt",truncation=True,max_length=512)
        with torch.inference_mode():
            ids=model.generate(**inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(target),
                max_new_tokens=256,num_beams=2,do_sample=False)
        output.append(tok.batch_decode(ids,skip_special_tokens=True)[0])
        progress.progress(min(counter[0]/max(total,1),1.0))
        live.text_area("Live translated chunks","\n\n".join(output[-8:]),
            height=150,key=f"live_{st.session_state.thread}_{counter[0]}")
    return " ".join(output)

def count_text(v):
    if isinstance(v,str): return len(chunks(v))
    if isinstance(v,dict): return sum(count_text(x) for x in v.values())
    if isinstance(v,list): return sum(count_text(x) for x in v)
    return 0

c1,c2=st.columns(2)
with c1: source=st.selectbox("📥 Source language",list(LANGS))
with c2: target=st.selectbox("📤 Target language",list(LANGS),index=1)

uploaded=st.file_uploader("📂 Upload PDF, DOCX, TXT or JSON",
                          type=["pdf","docx","txt","json"])
manual=st.text_area("✍️ Or enter text",height=100)
st.caption(f"Session ID: {st.session_state.thread}")

b1,b2=st.columns(2)
with b1: start=st.button("🚀 Translate Document",type="primary",use_container_width=True)
with b2: fresh=st.button("＋ New Session",use_container_width=True)

if fresh:
    st.session_state.thread=str(uuid.uuid4())
    st.session_state.result=None
    st.rerun()

if start:
    try:
        if source==target:
            st.warning("Choose different source and target languages.")
            st.stop()
        if uploaded: kind,data=read_file(uploaded)
        elif manual.strip(): kind,data="text",manual.strip()
        else:
            st.warning("Upload a file or enter text."); st.stop()

        tok,model=load_model()
        tok.src_lang=LANGS[source]
        progress=st.progress(0); status=st.empty(); live=st.empty()
        total=max(count_text(data),1); counter=[0]

        def walk(v):
            if isinstance(v,str):
                return translate_string(v,tok,model,LANGS[target],
                    progress,status,live,counter,total) if v.strip() else v
            if isinstance(v,list): return [walk(x) for x in v]
            if isinstance(v,dict): return {k:walk(x) for k,x in v.items()}
            return v

        translated=walk(data)
        json_result=translated if kind=="json" else None
        text=json.dumps(translated,ensure_ascii=False,indent=2) if kind=="json" else translated

        st.session_state.result={"text":text,"json":json_result,
                                 "source":source,"target":target}
        st.session_state.history.append({
            "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":source,"target":target,"thread":st.session_state.thread})
        progress.progress(1.0); status.success("✅ Translation completed!")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

if st.session_state.result:
    r=st.session_state.result
    st.markdown("---")
    st.subheader("✨ Translated Document")
    st.text_area("Final output",r["text"],height=220)
    d1,d2=st.columns(2)
    with d1:
        st.download_button("⬇ Download PDF",pdf_bytes(r["text"]),
            "translated.pdf","application/pdf",use_container_width=True)
        st.download_button("⬇ Download TXT",r["text"],
            "translated.txt","text/plain",use_container_width=True)
    with d2:
        st.download_button("⬇ Download DOCX",docx_bytes(r["text"]),
            "translated.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True)
        jout=r["json"] if r["json"] is not None else {
            "source_language":r["source"],"target_language":r["target"],
            "translation":r["text"]}
        st.download_button("⬇ Download JSON",
            json.dumps(jout,ensure_ascii=False,indent=2),
            "translated.json","application/json",use_container_width=True)

with st.expander("🕘 Translation History"):
    for item in reversed(st.session_state.history):
        st.write(f"{item['time']} | {item['source']} → {item['target']} | {item['thread']}")
