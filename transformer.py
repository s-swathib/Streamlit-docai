import transformers
from pyexpat import model
#from ctransformers import AutoModelForCausalLM
from langchain.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from time import time
import torch
from transformers import AutoTokenizer
import sys

def define_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# Mistral
# Load model directly
from transformers import AutoModel
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = define_tokenizer(model_name)
#EleutherAI/pythia-70m-deduped
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map="auto",)

llm = HuggingFacePipeline(pipeline=query_pipeline)

#loader = TextLoader("./archive/biden-sotu-2023-planned-official.txt",encoding="utf8")

loader= PyPDFDirectoryLoader("./llama-recipes/demo_apps/RAG_Chatbot_example/data")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

#retriever = vectordb.as_retriever()
from langchain.chains import ConversationalRetrievalChain

# Set up the Conversational Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(
             llm,
             vectordb.as_retriever(search_kwargs={'k': 2}),
             return_source_documents=True) 
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever, verbose=True)

# Start chatting with the chatbot
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa({'question': query, 'chat_history': chat_history})      
    print('Answer: ' + result['answer'] + '\n')   
    chat_history.append((query, result['answer']))    
