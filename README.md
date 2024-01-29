# Retrieval-Augmented-Generation-MultiLingual-Texts-
Here, we learn how to build a Retrieval-Augmented Generation (RAG) for multilingual texts. Since RAG works based on the information provided by vector stores, having the same vector standard for storing and retrieving data from vector databases is essential.
Suppose the documents stored in vector databases are in different languages (such as French or German). Then, the user writes a question in English and wants the answer. In such cases, there are two options: translators or prompts.
This code employs prompts to generate information from chunks of documents as questions and answers. We have also used ChoromaDB as our vector database to store data.

Imagine that you have many documents about companies in different languages and want to answer questions about specific items, such as the headquarters of a given company. Since the data template is repeated, you can not get a good result using RAG. It is due to many words that are in common between different sentences. 

