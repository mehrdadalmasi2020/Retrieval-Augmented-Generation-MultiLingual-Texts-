# Retrieval-Augmented-Generation-MultiLingual-Texts-
Here, we learn how to build a Retrieval-Augmented Generation (RAG) for multilingual texts. Since RAG works based on the information provided by vector stores, having the same vector standard for storing and retrieving data from vector databases is essential.
Suppose the documents stored in vector databases are in different languages (such as French or German). Then, the user writes a question in English and wants the answer. In such cases, there are two options: translators or prompts.
This code employs prompts to generate information from chunks of documents as questions and answers. We have also used ChoromaDB as our vector database to store data.

Imagine that you have many documents about a specific topic (e.g., different car manufacturers) in various languages. In such cases, the data template is repeated, and you can not get a good result using RAG. It is due to many words that are in common between different sentences. In other words, the only significant difference between different descriptions in such cases is the names of the companies.
If you try to answer questions such as the address of a given company's headquarters, LLM will likely receive an irrelevant results set. For such data, a good solution is to use Prompt instead of raw text. We can employ an efficient LLM to extract questions and answers from the dataset and store them in a vector datastore. 

1) Use prompt.py to generate the questions and answers and save them in a file.

2) Use cleansing.py to do the data cleansing on the generated file (removing irrelevant lines)

3) Use insert.py to insert data onto ChromaDB (a vector database)

4) Run RAG.py (based on Streamlit)


