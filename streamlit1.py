# Bring in deps
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
import sys

# set prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Embeddings into chroma DB
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                   model_kwargs={'device': 'cpu'})

#initialize the llm
llm=CTransformers(
    model="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type='llama',
    max_new_tokens=2048,
    top_k=10,
    top_p=0.9,
    temperature=0.0,
    repetition_penalty=1,
    context_length=4096,
)

#set data path and vector database path
data_path="./llama-recipes/demo_apps/RAG_Chatbot_example/data"
#DB_FAISS_PATH = 'vectorstore/db_faiss'

#load data and split to chunks
loader= PyPDFDirectoryLoader(data_path)
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
texts = text_splitter.split_documents(docs)
database = Chroma.from_documents(texts, embeddings)

retriever = database.as_retriever(search_kwargs={'k': 2})

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    database.as_retriever(search_kwargs={'k': 2}),
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