import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pinecone.init(api_key=PINECONE_API_KEY)


def ingest_document(filepath, source_url):
    loader = TextLoader(filepath)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = source_url

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Pinecone.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)


if __name__ == "__main__":
    # Example ingestion
    ingest_document("yourfile.txt", "https://yoursite.com/page")
