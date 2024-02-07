#import required libraries
import sys
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate

#set data path and vector database path
data_path="./llama-recipes/demo_apps/RAG_Chatbot_example/data"
DB_FAISS_PATH = 'vectorstore/db_faiss'

#load data and split to chunks
loader= PyPDFDirectoryLoader(data_path)
documents = loader.load()   
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# Embeddings into Faiss vactor DB
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                   model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(splits, embeddings)
db.save_local(DB_FAISS_PATH)  


#local lama2 llm using ctransformers
llm1=CTransformers(
    model="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type='llama',
    max_new_tokens=256,
    top_k=10,
    top_p=0.9,
    temperature=0.6,
    repetition_penalty=1,
    context_length=4096,
)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", model_type="llama",)

#create pipeline for llm and tokenizer
query_pipeline = transformers.pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
llm = HuggingFacePipeline(pipeline=query_pipeline)

#custom template for model response
custom_template = """
[INST]Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
{context}
Question: {question} [/INST]
"""

prompt = PromptTemplate(template=custom_template,
                        input_variables=["context", "question"],)

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    db.as_retriever(search_kwargs={'k': 2}),
    combine_docs_chain_kwargs={"prompt": prompt}    
    #return_source_documents=True
)


# Start chatting with the chatbot
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})      
    print('Answer: ' + result['answer'] + '\n') 
    
    chat_history.append((query, result['answer']))    
