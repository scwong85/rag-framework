import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import retrieve_context
from cache import Cache
import uvicorn
import openai
from openai import OpenAI
import time
import random
from config import Config
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import redis


USE_CLAUDE = os.getenv("USE_CLAUDE", "false").lower() == "true"
# === CONFIGURATION ===
EMBEDDING_DIM = 1536
REDIS_HOST = "localhost"
REDIS_PORT = 6379
VECTOR_INDEX = "semantic_cache"
openai.api_key = os.environ["OPENAI_API_KEY"]

cache = Cache()
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
openai_client = OpenAI()
redis_client = redis.from_url(Config.REDIS_URL)


class QuestionQuery(BaseModel):
    question: str


class TextInput(BaseModel):
    text: str


def get_llm():
    if Config.USE_CLAUDE:
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
        )
    else:
        return ChatOpenAI(
            model="gpt-4o", temperature=0, openai_api_key=Config.OPENAI_API_KEY
        )


def call_rag(summaries, question):
    retries = 0
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index = pc.Index(Config.PINECONE_INDEX_NAME)
    embeddings = OpenAIEmbeddings(
        openai_api_key=Config.OPENAI_API_KEY, model="text-embedding-3-small"
    )
    text_field = "text"
    vectorstore = PineconeVectorStore(index, embeddings, text_key=text_field)

    prompt_template = """Use the following pieces of context to answer the question at the end.  Try to answer in a structured way. Write your answer in HTML format but do not include ```html ```. Put words in bold that directly answer your question.
    If you don't know the answer, just say 'I am sorry I don't know the answer to this question.

    {summaries}


    Question: {question}.
    """

    PROMPT_WITH_SOURCE = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    qa_chain_source = RetrievalQAWithSourcesChain.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT_WITH_SOURCE},
        return_source_documents=True,
    )
    while retries < Config.MAX_RETRIES:
        try:
            response = qa_chain_source.invoke(
                {"question": question, "summaries": summaries}
            )
            # Extract sources as list of strings
            sources = []
            if "I don't know the answer" not in response["answer"]:
                source_docs = response["source_documents"]
                sources = [
                    doc.metadata.get("source", "")
                    for doc in source_docs
                    if doc.metadata.get("source")
                ]
            else:
                sources = []
            return {
                "question": question,
                "answer": response["answer"],
                "sources": list(set(sources)),  # List of URLs
            }
        except Exception as e:
            retries += 1
            time.sleep(2**retries + random.uniform(0, 1))
            if retries == Config.MAX_RETRIES and not Config.USE_CLAUDE:
                print("OpenAI failed; fallback to Claude")
                Config.USE_CLAUDE = True  # Soft switch
                retries = 0
    return {
        "question": question,
        "answer": "Sorry the service is temporarily unavailable",
        "sources": [],  # List of URLs
    }


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    ip = request.client.host
    count = redis_client.incr(ip)
    if count == 1:
        redis_client.expire(ip, 60)
    if count > 10:
        raise HTTPException(status_code=429, detail="Too many requests")
    return await call_next(request)


@app.middleware("https")
async def rate_limiter(request: Request, call_next):
    ip = request.client.host
    count = redis_client.incr(ip)
    if count == 1:
        redis_client.expire(ip, 60)
    if count > 10:
        raise HTTPException(status_code=429, detail="Too many requests")
    return await call_next(request)


@app.post("/ask")
async def ask(q: QuestionQuery):
    # semantic cache
    cache.create_index(VECTOR_INDEX, EMBEDDING_DIM)
    cached = cache.search_similar_question(q.question, VECTOR_INDEX)
    if cached:
        print("Cache found → returning cache")
        return cached

    print("Cache miss → calling RAG")
    context = retrieve_context(q.question)
    response = call_rag(context, q.question)
    cache.save_to_cache(q.question, response, VECTOR_INDEX)
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
