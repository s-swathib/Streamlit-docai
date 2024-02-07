# Bring in deps
import streamlit as st
import os
import tempfile
import re
from typing import Dict, List
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers

# Customize the layout
st.set_page_config(page_title="DOCAI", page_icon="🤖", layout="wide", )
#st.markdown(f"""<style>.stApp {{background-image: url("https://images.unsplash.com/photo-1509537257950-20f875b03669?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80");background-attachment: fixed;background-size: cover}}</style>""", unsafe_allow_html=True)

DB_FAISS_PATH = 'vectorstore/db_faiss'

# set prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Page:{page}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question","page"])
 
#template="Content: {page_content}\nSource: {source} \n Page:{page}", # look at the prompt does have page#
#                        input_variables=["page_content", "source","page"],)

# Embeddings into chroma DB
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                   model_kwargs={'device': 'cpu'})

#initialize the llm
llm=CTransformers(
    model="./models/mistral-7b-instruct-v0.1.Q8_0.gguf",
    model_type='mistral',
    max_new_tokens=512,
    top_k=10,
    top_p=0.9,
    temperature=0.0,
    repetition_penalty=1,
    context_length=4096,
)

st.title("📄 Document Conversation 🤖")
with st.sidebar:
    uploaded_files = st.file_uploader("Upload an article", accept_multiple_files=True)

def get_page_numbers(page_content: str) -> Dict[str, int]:
    match = re.search(r'Page \d+', page_content)
    if match:
        #return {"page": int(match.group()[5:])}
        return {"page": int(match.group(1))}
    return {}

for uploaded_file in uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=250,chunk_overlap=50, metadata_extractor=get_page_numbers)        
        texts = text_splitter.split_documents(docs)
        database = FAISS.from_documents(texts, embeddings)
        database.save_local(DB_FAISS_PATH)
        st.success("File Loaded Successfully!!")

        retriever = database.as_retriever(search_kwargs={'k': 2})

        # Query through LLM
        context=""
        question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?", disabled=not uploaded_file)
        if question:
            chunks = [question[i:i+510] for i in range(0, len(question), 510)]
            responses = []
            for chunk in chunks:
                context = texts
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, 
                    retriever=retriever, 
                    chain_type_kwargs={"prompt": prompt.format(context=context, question=question, page=page)})
                response = qa_chain.run(question)
                responses.append(response)
            st.info("\n\n".join(responses))
            docs = database.similarity_search(question)
            st.write(f"Query: {question}")
            st.write(f"Retrieved documents: {len(docs)}")
            for doc in docs:
                doc_details = doc.to_json()['kwargs']
                st.write("Source: ", doc_details['metadata']['source'])
                st.write("Text: ", doc_details['page_content'], "\n")
