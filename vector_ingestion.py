import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.redis import Redis
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

#Change the file name 
loader = TextLoader("filename.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=15)
docs = text_splitter.split_documents(documents)

# set your openAI api key as an environment variable
os.environ['OPENAI_API_KEY'] = ""

# OpenAI is used for embeddings please read for more details https://platform.openai.com/docs/guides/embeddings
embeddings = OpenAIEmbeddings(
model="text-embedding-3-small"
)

rds = Redis.from_documents(
docs,
embeddings,
#Specify the Redis URL
redis_url="",
#choose a name for your index we will use the same name in main.py
index_name="",
#Fill the password from your Redis cluster
password='')
