import json
import chromadb
import replicate
from chromadb import Documents
from chromadb.utils import embedding_functions
import sys
import os
import os
import time
from langchain.chains import MapReduceDocumentsChain, LLMChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.document_loaders import NewsURLLoader
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import torch
import gc
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_documents
 
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="microsoft/Orca-2-7b")
from chromadb.utils import embedding_functions


# Initialize Chromadb client
number_of_tables=0
client = chromadb.PersistentClient(path="/home/user/db")
collection = client.get_or_create_collection(
    name="test", embedding_function=sentence_transformer_ef
)

# Chromadb will convert a query string to embedding vectors used for similarity search.

doc=""
FinalAnswer=""
#for number_of_tables in range(0,1):


    # We will be searching for results that are similar to this string

query_string = st.text_input("Please enter your question here: ","Who is Gottlieb Daimler?")
print("sentence_transformer_ef: ",sentence_transformer_ef(query_string))

query_string1=query_string
question1=preprocess_documents([query_string1])[0]
str_question=""
for each_w in question1:
    str_question=str_question+" "+each_w
str_question=str_question.rstrip().lstrip().strip()
st.write("str_question: ",str_question)


# Perform the Chromadb query.
results = collection.query(
    query_texts=[str_question],#[query_string.replace("(","").replace(")","").replace("?","")],
    n_results=10,
)
st.write("the second step is started")
st.success('the second step is started!', icon="✅")

print(len(results["documents"]))
for e in results["documents"]:
    for con in e:
        doc=doc+con.rstrip().lstrip().strip()+" \n \n "
st.write(":green[the retrieved data from chromadb]")
st.write(":blue[=====================================================================]")
st.write(doc)
#del sentence_transformer_ef

from langchain.llms import CTransformers
config = {'max_new_tokens': 4096, 'temperature': 0.2, 'context_length': 8000}

try:
    del response
except:
    pass
try:
    del llm
except:
    pass
try:
    del llm_chain
except:
    pass
llm = CTransformers(model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config,threads=os.cpu_count(),gpu_layers=1)

template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words just from the context.
Answer the question below from the context below in several sentences. You must remove the unrelated information also. If the provided information is not related, you must say that you can not answer based on the provided information:  
{context}
{question} [/INST] </s>
"""

question_p = query_string#"""whcich company have a branch in Panama?"""
context_p = doc #""" On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""
prompt = PromptTemplate(template=template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.run({"question":question_p,"context":context_p})
#del llm_chain
#del llm
gc.collect()
torch.cuda.empty_cache()  
#    FinalAnswer=FinalAnswer+response+" \n ***************** \n"
st.success('the third step is started!', icon="✅")

st.write(":blue[=====================================================================]")
st.write(":orange[Answer:]")
st.write(response)

