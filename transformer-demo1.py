from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


time_1 = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_2 = time()
print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")

time_1 = time()
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
time_2 = time()
print(f"Prepare pipeline: {round(time_2-time_1, 3)} sec.")

def test_model(tokenizer, pipeline, prompt_to_test):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        prompt_to_test: the prompt
    Returns
        None
    """
    # adapted from https://huggingface.co/blog/llama2#using-transformers
    time_1 = time()
    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,)
    time_2 = time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

#test_model(tokenizer,query_pipeline,"What were the main topics in the State of the Union in 2023? Summarize. Keep it under 200 words.")

llm = HuggingFacePipeline(pipeline=query_pipeline)
# checking again that everything is working fine
#llm(prompt="Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words.")

loader = TextLoader("archive/biden-sotu-2023-planned-official.txt",encoding="utf8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

def test_rag(qa, query):
    #print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)

query = input("Enter the query: ")
test_rag(qa, query)
