
import io,json,re,time,uuid,importlib
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

MODEL="olob0/nllb-200-distilled-600M-ct2-int8_float16"
LANG={"English":"eng_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Tamil":"tam_Taml","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Urdu":"urd_Arab","French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn","Italian":"ita_Latn","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan"}

st.set_page_config(page_title="Document Translation",page_icon="🌐",layout="centered")
st.markdown("""<style>
.stApp{background:#f4f7fc;color:#18243b}
.block-container{max-width:900px;padding:1.5rem 1rem 3rem}
header[data-testid="stHeader"]{background:#f4f7fc}
.hero{background:linear-gradient(120deg,#071d43,#124b89);padding:25px 16px;border-radius:16px;text-align:center;margin-bottom:20px;box-shadow:0 6px 20px #102c501c}
.hero h1{color:#fff;font-size:clamp(23px,4vw,32px);font-weight:800;margin:0}
.hero span{color:#ffc928}
.hero p{color:#e0eaff;font-size:14px;margin:8px 0 0}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:14px}
.stButton>button,.stDownloadButton>button{background:#ffc928;color:#10213d;border:0;border-radius:9px;font-weight:700;min-height:42px}
.stButton>button:hover,.stDownloadButton>button:hover{background:#ffdb69;color:#071d43;border:1px solid #124b89}
.stProgress>div>div>div>div{background:#124b89}
textarea,input{border-radius:9px!important}
.live{background:#fff;border:1px solid #d7e2f3;border-left:5px solid #124b89;padding:16px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere;min-height:90px;color:#17243b}
div[data-testid="stFileUploader"]{background:#fff;border:1px dashed #9eb5d5;border-radius:12px;padding:10px}
small{color:#62718a}
</style>
<div class="hero"><h1><span>Document</span> Translation</h1><p>Multilingual documents · Live chunk output · Downloadable results</p></div>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading translation model...")
def load():
    ct=importlib.import_module("ctranslate2")
    path=snapshot_download(MODEL)
    return AutoTokenizer.from_pretrained(path),ct.Translator(path,device="cpu",compute_type="int8",inter_threads=1,intra_threads=2)

def read(f):
    b,n=f.getvalue(),f.name.lower()
    if n.endswith(".pdf"): return "text","\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(b)).pages)
    if n.endswith(".docx"): return "text","\n".join(p.text for p in Document(io.BytesIO(b)).paragraphs)
    if n.endswith(".json"): return "json",json.loads(b.decode("utf-8-sig"))
    return "text",b.decode("utf-8-sig",errors="replace")

def split(s,limit=700):
    out=[]
    for line in s.splitlines():
        if not line.strip(): out.append(""); continue
        cur=""
        for p in re.split(r'(?<=[.!?।؟])\s+',line.strip()):
            while len(p)>limit:
                if cur: out.append(cur); cur=""
                out.append(p[:limit]); p=p[limit:]
            if cur and len(cur)+len(p)+1>limit: out.append(cur); cur=p
            else: cur=(cur+" "+p).strip()
        if cur: out.append(cur)
    return [x for x in out if x.strip()]

def translate(s,src,tgt,update):
    if not s.strip(): return s
    tok,model=load(); tok.src_lang=src
    ids=tok(s,truncation=True,max_length=256).input_ids
    source=tok.convert_ids_to_tokens(ids); generated=[]
    def dec(ts):
        ts=[x for x in ts if x!=tgt]
        return tok.decode(tok.convert_tokens_to_ids(ts),skip_special_tokens=True)
    def cb(step):
        if step.token!=tgt: generated.append(step.token)
    try:
        f=model.translate_batch([source],target_prefix=[[tgt]],beam_size=1,max_decoding_length=300,callback=cb,asynchronous=True)[0]
        last=-1
        while not f.done():
            if len(generated)!=last: update(dec(generated)); last=len(generated)
            time.sleep(.05)
        result=dec(f.result().hypotheses[0])
    except TypeError:
        ts=model.translate_batch([source],target_prefix=[[tgt]],beam_size=2,max_decoding_length=300)[0].hypotheses[0]
        if ts and ts[0]==tgt: ts=ts[1:]
        result=tok.decode(tok.convert_tokens_to_ids(ts),skip_special_tokens=True)
    update(result); return result

def walk(x,fn):
    if isinstance(x,str): return fn(x) if x.strip() else x
    if isinstance(x,list): return [walk(v,fn) for v in x]
    if isinstance(x,dict): return {k:walk(v,fn) for k,v in x.items()}
    return x

def make_docx(s):
    d=Document()
    for line in s.splitlines(): d.add_paragraph(line)
    b=io.BytesIO(); d.save(b); return b.getvalue()

def make_pdf(s):
    p=FPDF(); p.set_auto_page_break(True,15); p.add_page(); p.set_font("Helvetica",size=10)
    for line in s.splitlines(): p.multi_cell(0,6,line.encode("latin-1","replace").decode("latin-1"))
    return bytes(p.output())

if "thread" not in st.session_state: st.session_state.thread="THR-"+uuid.uuid4().hex[:8].upper()
with st.container(border=True):
    a,b=st.columns([3,1])
    a.text_input("Session ID",st.session_state.thread,disabled=True)
    if b.button("New session",use_container_width=True):
        st.session_state.thread="THR-"+uuid.uuid4().hex[:8].upper()
        st.session_state.pop("result",None); st.rerun()
    a,b=st.columns(2)
    src=a.selectbox("Source language",list(LANG))
    tgt=b.selectbox("Target language",list(LANG),index=2)
    f=st.file_uploader("Upload a document",type=["pdf","docx","txt","json"])
    pasted=st.text_area("Or paste text",height=110,placeholder="Enter text to translate…")
    st.caption("Supported: PDF, DOCX, TXT and JSON. Long documents are processed in chunks.")
    st.subheader("Live output")
    live=st.empty()
    live.markdown('<div class="live">Your current translation chunk will appear here.</div>',unsafe_allow_html=True)
    if st.button("Translate document",type="primary",use_container_width=True):
        if src==tgt: st.warning("Select different source and target languages.")
        elif not f and not pasted.strip(): st.warning("Upload a document or enter text.")
        else:
            try:
                kind,data=read(f) if f else ("text",pasted)
                chunks=[]
                if kind=="json":
                    def gather(x):
                        if isinstance(x,str) and x.strip(): chunks.append(x)
                        elif isinstance(x,(dict,list)):
                            for v in (x.values() if isinstance(x,dict) else x): gather(v)
                    gather(data); chunks=list(dict.fromkeys(chunks))
                else: chunks=split(data)
                if not chunks: st.error("No readable text found."); st.stop()
                bar=st.progress(0,text=f"Preparing {len(chunks)} chunks…"); results={}
                for i,s in enumerate(chunks,1):
                    def update(part,idx=i):
                        safe=part.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        live.markdown(f'<div class="live"><b>Chunk {idx}/{len(chunks)}</b><hr>{safe or "Generating…"}</div>',unsafe_allow_html=True)
                    results[s]=translate(s,LANG[src],LANG[tgt],update)
                    bar.progress(i/len(chunks),text=f"Translated {i}/{len(chunks)} chunks")
                obj=walk(data,lambda s:results.get(s,s)) if kind=="json" else "\n".join(results[s] for s in chunks)
                output=json.dumps(obj,ensure_ascii=False,indent=2) if kind=="json" else obj
                meta=obj if kind=="json" else {"session_id":st.session_state.thread,"source_language":src,"target_language":tgt,"translated_text":output}
                st.session_state.result={"text":output,"json":meta}
                bar.empty(); live.markdown('<div class="live"><b>Translation complete</b><br>Your complete result is ready below.</div>',unsafe_allow_html=True)
                st.success("Translation completed successfully.")
            except Exception as e: st.error(f"Translation failed: {e}")

r=st.session_state.get("result")
if r:
    st.subheader("Download translated document")
    c=st.columns(4)
    c[0].download_button("📄 PDF",make_pdf(r["text"]),"translation.pdf","application/pdf",use_container_width=True)
    c[1].download_button("📝 TXT",r["text"].encode(),"translation.txt","text/plain",use_container_width=True)
    c[2].download_button("📑 DOCX",make_docx(r["text"]),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
    c[3].download_button("📋 JSON",json.dumps(r["json"],ensure_ascii=False,indent=2).encode(),"translation.json","application/json",use_container_width=True)
