# !pip install gensim
import chromadb
import json
from chromadb import Documents
import chromadb
import sys
import os
import replicate
import torch
import gc
from tokenizers import Tokenizer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_documents
import chromadb
from chromadb.config import Settings
import json
import chromadb
import replicate
from chromadb import Documents
from vectorhub.encoders.text.tfhub import Bert2Vec
from vectorhub.bi_encoders.qa.torch_transformers import DPR2Vec
from chromadb.utils import embedding_functions

gc.collect()
torch.cuda.empty_cache()

# create a database
client = chromadb.PersistentClient(path="/home/user/db")
path='/mnt/data/'
# create a table
collection = client.get_or_create_collection(name="test")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="microsoft/Orca-2-7b")
print("model loaded")


for data in Clean('/mnt/data/'+"/try/"+"test.txt"):  #CleanText has been produced in the cleansing.py file.
    
    doc=data.split("\n")
    documents=" "
    head=""
    for eachline in doc:
        if "?" not in eachline:
            documents=documents+eachline+"\n\n"
        else:
            documents=documents+eachline
    documents=documents.rstrip().lstrip().strip()
  
    splits=documents.split("\n\n")
        
    for newline_index in range(0,len(splits)):
        splits[newline_index]=splits[newline_index].replace("\n"," ").replace("?","?\n").rstrip().lstrip().strip()
        
        
    try:
        splits.remove("\n")
        splits.remove(" \n")
        
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()            


    import fnmatch
    c=0
    for v in splits:
        tempsplits=[]

        newv=v.split(') \n')[0].replace("(","").replace(")","")
        if " Answer" in newv:
            newv=newv.split(" Answer")[0]
        if "Answer " in newv:
            newv=newv.split("Answer ")[0]
        
        if ".  A" in newv:
            newv=newv.split(".  A")[0]
        if "A:" in newv:
            newv=newv.split("A: ")[0]

        # Note that only the question part is used to determine the word embedding vectors
      
        newv=newv.rstrip().lstrip().strip()
        import re
        question=preprocess_documents([newv])[0]
        str_question=""
        for each_w in question:
            str_question=str_question+" "+each_w
        str_question=str_question.replace("Question "," ").replace("question "," ")
        tempsplits.append(str_question)
        output=sentence_transformer_ef(tempsplits)

        each=v
        dataset=[]
        if len(each)>10 and "?" in each :
            dataset.append({"id":str(c), "description":each})
            gc.collect()
            torch.cuda.empty_cache()
            collection.upsert(
                embeddings = output,
                documents = [ dataset[0]["description"]],
                metadatas = [{"source": head}],
                ids = [str(count)]

            )
            print("done: ")
        c=c+1

    del splits
    del output
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")



