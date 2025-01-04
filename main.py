import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from uuid import uuid4
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Load environment variables
load_dotenv()   
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI
chat = ChatOpenAI(
    model='gpt-3.5-turbo'
)

# Initialize Pinecone
pc = Pinecone(
    api_key="pcsk_2Yu1PZ_TcpEUcUm3eng2CsCqoTHU6PnCj4ENqvzrn9dqz7ovp7brdrDSCj3PsvUkt4d4s"  # Move API key to environment variables
)

index_name = 'ragg'

# Check if index exists and create if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        ),
        dimension=1536
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Get the index
index = pc.Index(index_name)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize vector store correctly
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# Process PDF
pdfreader = PdfReader('data.pdf')
raw_text = ''

for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)

texts = text_splitter.split_text(raw_text)

# Convert text chunks to Document objects
documents = [Document(page_content=text) for text in texts]

# print(documents)

uuids = [str(uuid4()) for _ in range(len(documents))]

# Add documents to vector store
vector_store.add_documents(documents=documents, ids=uuids)

def query_vectorstore(query: str, k: int = 3):
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"Error during query: {str(e)}")
        return None


print("Test query:")
# Test query
try:
    results = query_vectorstore("tell me about the Post-Vietnam and Watergate")
    if results:
        print("Query results:")
        for doc in results:
            print("\nDocument content:", doc.page_content)
except Exception as e:
    print(f"Error during query: {str(e)}")