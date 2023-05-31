import os
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI, VectorDBQA
import config

def run_question_query(input):
    txt = Path('knowledge/totem_knowledge.txt').read_text(encoding="utf8")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0 )
    texts = text_splitter.split_text(txt)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)

    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectorstore)

    answer = qa.run(input)
    return answer
