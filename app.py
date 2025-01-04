import os
import time
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
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
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model='gpt-3.5-turbo'
)

# Initialize chat messages
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hi AI, How are you today?"),
    AIMessage(content="I am great thank you, How can I help you?"),
    HumanMessage(content="What is your role")
]

res = chat(messages)
messages.append(res)

prompt = HumanMessage(content="Can you help me with my assignments?")
messages.append(prompt)
res = chat(messages)

# Initialize Pinecone
pc = Pinecone(
    api_key='pcsk_2Yu1PZ_TcpEUcUm3eng2CsCqoTHU6PnCj4ENqvzrn9dqz7ovp7brdrDSCj3PsvUkt4d4s',  
    environment='us-east-1' 
)

all_indexes = pc.list_indexes()
print("all_indexes", all_indexes)
# pc.delete_index("llama-rag-2")

index_name = 'llama-rag-2'
new_index_name = 'ragg'

# Check if index exists and create if it doesn't
if new_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=new_index_name,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        ),
        dimension=1536
    )

    while not pc.describe_index(new_index_name).status['ready']:
        time.sleep(1)

# Connect to the index
index = pc.Index(new_index_name)
print(f"Connected to index: {index}")

# Create embeddings
embeddings = OpenAIEmbeddings()


vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)


# Function to add texts to the vector store
def add_texts_to_vectorstore(texts):
    vector_store.add_texts(texts)

# Function to query the vector store
def query_vectorstore(query: str, k: int = 3):
    # Get the embedding for the query
    results = vector_store.similarity_search(
        query,
        k=k
    )
    return results

# Example usage for adding texts
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
print(texts)

# Add texts to vector store
add_texts_to_vectorstore(texts)


testing_retrieval = query_vectorstore("tell me about the Post-Vietnam and Watergate")
print("Querying vector store", )

try:
    results = query_vectorstore("tell me about the Post-Vietnam and Watergate")
    print("Query results:")
    for doc in results:
        print("\nDocument content:", doc.page_content)
except Exception as e:
    print(f"Error during query: {str(e)}")