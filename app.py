
import io,json,re,uuid,time,hashlib,importlib,html
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

MODEL="olob0/nllb-200-distilled-600M-ct2-int8_float16"
LANG={"English":"eng_Latn","Hindi":"hin_Deva","Telugu":"tel_Telu","Tamil":"tam_Taml","Kannada":"kan_Knda","Malayalam":"mal_Mlym","Bengali":"ben_Beng","Marathi":"mar_Deva","Urdu":"urd_Arab","French":"fra_Latn","German":"deu_Latn","Spanish":"spa_Latn","Italian":"ita_Latn","Arabic":"arb_Arab","Chinese":"zho_Hans","Japanese":"jpn_Jpan"}

def clean(text,max_phrase=8):
    words=text.split();out=[];i=0
    norm=lambda s:re.sub(r"^\W+|\W+$","",s).casefold()
    while i<len(words):
        matched=False
        for n in range(min(max_phrase,(len(words)-i)//3),0,-1):
            phrase=[norm(x) for x in words[i:i+n]]
            if not all(phrase):continue
            count=1
            while i+(count+1)*n<=len(words) and [norm(x) for x in words[i+count*n:i+(count+1)*n]]==phrase:count+=1
            if count>=3:
                out.extend(words[i:i+n]);i+=count*n;matched=True;break
        if not matched:out.append(words[i]);i+=1
    return " ".join(out)

st.set_page_config(page_title="AIT GLOBAL TRANSLATOR",page_icon="🌐",layout="centered")
st.markdown("""<style>.stApp{background:#f4f7fc;color:#18243b}.block-container{max-width:900px;padding:1rem}.hero{background:linear-gradient(120deg,#071d43,#124b89);padding:22px;border-radius:16px;text-align:center}.hero h1{color:white}.hero span{color:#ffc928}.hero p{color:#e0eaff}.stButton>button,.stDownloadButton>button{background:#ffc928;color:#10213d;border:0;border-radius:9px;font-weight:bold}.live{background:white;border:1px solid #d7e2f3;border-left:5px solid #124b89;padding:12px;border-radius:9px;white-space:pre-wrap;overflow-wrap:anywhere}</style><div class="hero"><h1><span>AIT GLOBAL</span> TRANSLATOR</h1><p>Live Streaming · Chunk Cache · Document Downloads</p></div>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading translation model...")
def load():
    ct=importlib.import_module("ctranslate2");path=snapshot_download(MODEL)
    return AutoTokenizer.from_pretrained(path),ct.Translator(path,device="cpu",compute_type="int8",inter_threads=1,intra_threads=2)

def read_file(f):
    b,name=f.getvalue(),f.name.lower()
    if name.endswith(".pdf"):return "text","\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(b)).pages)
    if name.endswith(".docx"):return "text","\n".join(p.text for p in Document(io.BytesIO(b)).paragraphs)
    if name.endswith(".json"):return "json",json.loads(b.decode("utf-8-sig"))
    return "text",b.decode("utf-8-sig",errors="replace")

def split_text(text,tok,limit=300):
    chunks=[];current=""
    for sentence in re.split(r'(?<=[.!?।؟])\s+|\n+',text.strip()):
        for word in sentence.split():
            candidate=(current+" "+word).strip()
            if current and len(tok(candidate).input_ids)>limit:
                chunks.append(current);current=word
            else:current=candidate
    if current:chunks.append(current)
    return chunks

def translate(text,src,tgt,tok,model,update):
    tok.src_lang=src
    source=tok.convert_ids_to_tokens(tok(text).input_ids);generated=[]
    def callback(step):
        if step.token!=tgt:generated.append(step.token)
    def decode(tokens):
        return clean(tok.decode(tok.convert_tokens_to_ids([x for x in tokens if x!=tgt]),skip_special_tokens=True))
    try:
        future=model.translate_batch([source],target_prefix=[[tgt]],beam_size=1,max_decoding_length=512,callback=callback,asynchronous=True)[0]
        last=-1
        while not future.done():
            if len(generated)!=last:update(decode(generated));last=len(generated)
            time.sleep(.05)
        return decode(future.result().hypotheses[0]).strip()
    except (TypeError,AttributeError):
        return decode(model.translate_batch([source],target_prefix=[[tgt]],beam_size=2,max_decoding_length=512)[0].hypotheses[0]).strip()

def walk(obj,fn):
    if isinstance(obj,str):return fn(obj) if obj.strip() else obj
    if isinstance(obj,list):return [walk(x,fn) for x in obj]
    if isinstance(obj,dict):return {k:walk(v,fn) for k,v in obj.items()}
    return obj

def make_docx(text):
    doc=Document()
    for line in text.splitlines():doc.add_paragraph(line)
    buf=io.BytesIO();doc.save(buf);return buf.getvalue()

def make_pdf(text):
    pdf=FPDF();pdf.set_auto_page_break(True,15);pdf.add_page();pdf.set_font("Helvetica",size=10)
    for line in text.splitlines():pdf.multi_cell(0,6,line.encode("latin-1","replace").decode("latin-1"))
    return bytes(pdf.output())

if "thread" not in st.session_state:st.session_state.thread="THR-"+uuid.uuid4().hex[:8].upper()
if "cache" not in st.session_state:st.session_state.cache={}
if "result" not in st.session_state:st.session_state.result=None

a,b=st.columns([3,1]);a.text_input("Session ID",st.session_state.thread,disabled=True)
if b.button("New session",use_container_width=True):
    st.session_state.thread="THR-"+uuid.uuid4().hex[:8].upper();st.session_state.cache={};st.session_state.result=None;st.rerun()

a,b=st.columns(2);src=a.selectbox("Source language",LANG);tgt=b.selectbox("Target language",LANG,index=2)
file=st.file_uploader("Upload PDF, DOCX, TXT or JSON",type=["pdf","docx","txt","json"])
pasted=st.text_area("Or paste text",height=100);live=st.empty();slot=st.empty()

if st.button("Translate document",type="primary",use_container_width=True):
    if src==tgt:st.warning("Choose different languages.")
    elif not file and not pasted.strip():st.warning("Upload a file or enter text.")
    else:
        try:
            kind,data=read_file(file) if file else ("text",pasted)
            tok,model=load();completed=[]
            def show(label,partial=""):
                text="\n\n".join(completed+([clean(partial)] if partial else []))
                live.markdown('<div class="live"><b>'+html.escape(label)+'</b><hr>'+html.escape(text)+'</div>',unsafe_allow_html=True)
            if kind=="json":
                strings=[]
                def gather(x):
                    if isinstance(x,str) and x.strip():strings.append(x)
                    elif isinstance(x,(dict,list)):
                        for v in (x.values() if isinstance(x,dict) else x):gather(v)
                gather(data);unique=list(dict.fromkeys(strings));mapped={};bar=slot.progress(0)
                for i,s in enumerate(unique,1):
                    key=hashlib.sha256((MODEL+src+tgt+s).encode()).hexdigest()
                    if key not in st.session_state.cache:
                        st.session_state.cache[key]=translate(s,LANG[src],LANG[tgt],tok,model,lambda p,j=i:show(f"JSON {j}/{len(unique)}",p))
                    mapped[s]=clean(st.session_state.cache[key]);completed.append(mapped[s])
                    show(f"JSON field {i}/{len(unique)}");bar.progress(i/max(len(unique),1))
                json_result=walk(data,lambda s:mapped.get(s,s))
                output=json.dumps(json_result,ensure_ascii=False,indent=2)
            else:
                pieces=split_text(data,tok)
                if not pieces:st.error("No readable text found.");st.stop()
                bar=slot.progress(0)
                for i,piece in enumerate(pieces,1):
                    key=hashlib.sha256((MODEL+src+tgt+piece).encode()).hexdigest()
                    if key not in st.session_state.cache:
                        st.session_state.cache[key]=translate(piece,LANG[src],LANG[tgt],tok,model,lambda p,j=i:show(f"Chunk {j}/{len(pieces)}",p))
                    completed.append(clean(st.session_state.cache[key]))
                    show(f"Chunk {i}/{len(pieces)} completed");bar.progress(i/len(pieces))
                output="\n\n".join(completed)
                json_result={"session_id":st.session_state.thread,"source_language":src,"target_language":tgt,"translated_text":output}
            output=clean(output);st.session_state.result={"text":output,"json":json_result}
            show("Translation complete");st.success("Translation completed.")
        except Exception as e:st.error(f"Translation failed: {e}")

r=st.session_state.result
if r:
    st.subheader("Download translated document");c=st.columns(4)
    c[0].download_button("📄 PDF",make_pdf(r["text"]),"translation.pdf","application/pdf",use_container_width=True)
    c[1].download_button("📝 TXT",r["text"].encode("utf-8"),"translation.txt","text/plain",use_container_width=True)
    c[2].download_button("📑 DOCX",make_docx(r["text"]),"translation.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
    c[3].download_button("📋 JSON",json.dumps(r["json"],ensure_ascii=False,indent=2).encode("utf-8"),"translation.json","application/json",use_container_width=True)
