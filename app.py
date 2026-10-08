import io, uuid, torch, chromadb, streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

MODEL="facebook/nllb-200-distilled-600M"
LANG={"English":"eng_Latn","Tamil":"tam_Taml","Telugu":"tel_Telu","Hindi":"hin_Deva",
"Kannada":"kan_Knda","Malayalam":"mal_Mlym","French":"fra_Latn",
"German":"deu_Latn","Spanish":"spa_Latn"}

# SYSTEM GENERATED THREAD ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id=str(uuid.uuid4())

@st.cache_resource
def load_model():
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL,low_cpu_mem_usage=True)
    model.eval()
    return tokenizer,model

@st.cache_resource
def get_db():
    client=chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("translation_history")

def read_file(f):
    if not f:return ""
    data=f.getvalue()
    ext=f.name.lower().split(".")[-1]
    if ext=="txt":return data.decode("utf-8",errors="ignore")
    if ext=="pdf":return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if ext=="docx":return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return ""

def translate(text,src,tgt):
    tokenizer,model=load_model()
    tokenizer.src_lang=src
    target_id=tokenizer.convert_tokens_to_ids(tgt)
    parts=[text[i:i+1200] for i in range(0,len(text),1200)]
    result=[]
    bar=st.progress(0)

    for i,part in enumerate(parts):
        if not part.strip():continue
        inputs=tokenizer(part,return_tensors="pt",truncation=True,max_length=96)

        with torch.inference_mode():
            output=model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False
            )

        result.append(tokenizer.batch_decode(output,skip_special_tokens=True)[0])
        bar.progress((i+1)/len(parts))

    bar.empty()
    return " ".join(result)

st.title("🌐 AI Translator")
st.caption("Hugging Face NLLB-200 + ChromaDB")

c1,c2=st.columns(2)
src=c1.selectbox("Source Language",list(LANG))
tgt=c2.selectbox("Target Language",list(LANG),index=1)

text=st.text_area("Text to translate",height=150)
file=st.file_uploader("📁 Upload TXT / PDF / DOCX",type=["txt","pdf","docx"])

if st.button("🚀 Translate",type="primary",use_container_width=True):
    original=read_file(file) if file else text

    if not original.strip():
        st.warning("Enter text or upload a file.")
    elif src==tgt:
        st.warning("Select different languages.")
    else:
        try:
            with st.spinner("Translating..."):
                result=translate(original,LANG[src],LANG[tgt])

            # SAVE TRANSLATION + SYSTEM THREAD ID
            get_db().add(
                ids=[str(uuid.uuid4())],
                documents=[result],
                metadatas=[{
                    "source":src,
                    "target":tgt,
                    "original":original,
                    "thread_id":st.session_state.thread_id
                }]
            )

            st.success("✅ Translation completed")
            st.text_area("Translation Result",result,height=220)
            st.download_button("📥 Download",result,"translation.txt")

        except Exception as e:
            st.error(f"Translation Error: {e}")

st.divider()

st.write("🧵 **Current Thread ID:**")
st.code(st.session_state.thread_id)

if st.button("＋ New Thread",use_container_width=True):
    st.session_state.thread_id=str(uuid.uuid4())
    st.rerun()

st.divider()
st.subheader("🕘 Translation History")

search=st.text_input("🔍 Search History")

data=get_db().get(include=["documents","metadatas"])

for doc,meta in zip(
    data.get("documents",[])[::-1],
    data.get("metadatas",[])[::-1]
):
    if not search or search.lower() in (doc+str(meta)).lower():
        with st.expander(
            f"🌐 {meta.get('source')} → {meta.get('target')} | "
            f"Thread: {meta.get('thread_id')}"
        ):
            st.write("**Original:**",meta.get("original",""))
            st.write("**Translation:**",doc)

if st.button("🗑️ Clear All History"):
    ids=get_db().get()["ids"]
    if ids:
        get_db().delete(ids=ids)
    st.success("History cleared.")
    st.rerun()