import os

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

# Load and process the text
loader = TextLoader('state_of_the_union.txt')
documents = loader.load()

# This is a parameter that you can tune to take a different chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)