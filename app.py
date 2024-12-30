import os
from langchain import chat_models
from langchain.chat_models import ChatOpenAI
from datasets import load_dataset
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

load_dotenv()



os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    openai_api_key = os.environ['OPENAI_API_KEY'],
    model = 'gpt-3.5-turbo'
)

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content="Hi AI, How are you today?"),
    AIMessage(content="I am great thank you, How can I help you?"),
    HumanMessage(content="What is your role")
]

res = chat(messages)


messages.append(res)

prompt = HumanMessage(
    content = "Can you help mw with my assignments?"
)

messages.append(prompt)

res = chat(messages)



langchain_information = [
    "An LLM (Large Language Model) Chain is a concept in the context of working with large language models, like OpenAI's GPT or similar, for building sophisticated pipelines that involve multiple steps of processing, reasoning, and decision-making. It is often implemented using frameworks like LangChain, which provides tools to integrate LLMs with external tools, memory, databases, and other capabilities"
]


source_knowledge = langchain_information



query = "Can you tell me about the LLMchain in langchain?"

augmented_prompt = f"""Using the contexts below, answer the query. {source_knowledge} Query: {query}"""


pc = Pinecone(
    api_key='pcsk_2Yu1PZ_TcpEUcUm3eng2CsCqoTHU6PnCj4ENqvzrn9dqz7ovp7brdrDSCj3PsvUkt4d4s',  
    environment='us-east-1' 
)

import time 
index_name = 'llama-rag-2'


new_index_name = 'ragg'


if index_name not in pc.list_indexes():
    pc.create_index(
        name = new_index_name,
        metric='euclidean',
        spec=ServerlessSpec(
             cloud='aws',
            region='us-east-1'
        ),
         dimension=1536
         
    )

    while not Pinecone.describe_index(name = new_index_name).status['ready']:
            time.sleep(1)

    index = Pinecone.Index(index_name)

    print(index)

pdfreader  = PdfReader('data.pdf')

raw_text = ''

for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = RecursiveCharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len
)
