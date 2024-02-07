import streamlit as st
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import CTransformers
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Customize the layout
st.set_page_config(page_title="DOCAI", page_icon="🤖", layout="wide", )

#Title for app
st.title('LLM Model- (Question-Answering) with private data')

#upload multiple files
with st.sidebar:
    uploaded_files = st.file_uploader("Upload files..", accept_multiple_files=True)

data_path="./llama-recipes/demo_apps/RAG_Chatbot_example/data"
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Embeddings into Faiss vactor DB
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                   model_kwargs={'device': 'cpu'})


loader= PyPDFDirectoryLoader(data_path)
documents = loader.load()  
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = text_splitter.split_documents(documents)
db = FAISS.from_documents(splits, embeddings)
db.save_local(DB_FAISS_PATH)
st.success("File Loaded Successfully!!")

llm=CTransformers(
    model="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type='llama',
    max_new_tokens=512,
    top_k=10,
    top_p=0.9,
    temperature=0.0,
    repetition_penalty=1,
    context_length=4096,
)

custom_template = """
[INST]Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
{context}
Question: {question} [/INST]
"""

prompt = PromptTemplate(template=custom_template,
                        input_variables=["context", "question"],)

retriever=db.as_retriever(search_kwargs={'k': 2})

def generate_response(input_text):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate(
            template=custom_template,
            input_variables=["context", "question"],
            ),
        }
    )    
    st.info(qa_chain.run(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What is prompting?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)