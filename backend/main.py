import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import retrieve_context
from cache import Cache
import uvicorn
import openai
from openai import OpenAI
import anthropic
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
import json
import numpy as np
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


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


def embed(text):
    res = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(res.data[0].embedding, dtype=np.float32)


# === REDIS VECTOR INDEX SETUP ===
def create_index():
    try:
        redis_client.ft(VECTOR_INDEX).info()
    except:
        print("Index not found, creating...")
        redis_client.ft(VECTOR_INDEX).create_index(
            fields=[
                TextField("question"),
                TextField("answer"),
                VectorField("embedding", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": EMBEDDING_DIM,
                    "DISTANCE_METRIC": "COSINE",
                })
            ],
            definition=IndexDefinition(prefix=["q:"], index_type=IndexType.HASH)
        )


# === SEARCH CACHE ===
def search_similar_question(question, threshold=0.2):
    print("searching for answer for question", question)
    vector = embed(question).tobytes()
    q = Query("*=>[KNN 1 @embedding $vec_param AS score]") \
        .sort_by("score") \
        .return_fields("question", "answer", "score") \
        .dialect(2)
    params = {"vec_param": vector}

    res = redis_client.ft(VECTOR_INDEX).search(q, query_params=params)
    print("response from cache")
    print(res)
    if res.total > 0 and float(res.docs[0].score) < threshold:
        print("cached matched")
        return json.loads(res.docs[0].answer)
    return None


# === SAVE TO CACHE ===
def save_to_cache(question: str, answer: dict):
    vector = embed(question).tobytes()
    # redis_client.hset(
    #     f"q:{question}",
    #     mapping={
    #         "question": question,
    #         "answer": json.dumps(answer),
    #         "embedding": vector
    #     }
    # )
    redis_client.ft(VECTOR_INDEX).add_document(
        f"q:{question}",
        replace=True,
        question=question,
        answer=json.dumps(answer),
        embedding=vector
    )


def call_model(context, question):
    retries = 0
    while retries < Config.MAX_RETRIES:
        try:
            if Config.USE_CLAUDE:
                client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQ: {question}",
                        }
                    ],
                )
                return response.content[0].text
            else:
                openai.api_key = Config.OPENAI_API_KEY
                messages = [
                    {"role": "system", "content": f"Use this context: {context}"},
                    {"role": "user", "content": question},
                ]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages, max_tokens=500
                )
                return response["choices"][0]["message"]["content"]
        except Exception as e:
            retries += 1
            time.sleep(2**retries + random.uniform(0, 1))
            if retries == Config.MAX_RETRIES and not Config.USE_CLAUDE:
                print("OpenAI failed; fallback to Claude")
                Config.USE_CLAUDE = True  # Soft switch
                retries = 0
    return "Sorry, service is temporarily unavailable."


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
    If you don't know the answer, just say 'I am sorry I dont know the answer to this question or you dont have access to the files needed to answer the question.' Don't try to make up an answer.

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
            if "I dont know the answer" not in response["answer"]:
                source_docs = response["source_documents"]
                sources = [
                    doc.metadata.get("source", "")
                    for doc in source_docs
                    if doc.metadata.get("source")
                ]
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


class TextInput(BaseModel):
    text: str


@app.middleware("http")
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
    # cached = cache.get(q.question)
    # if cached:
    #     print("Getting answer from cache")
    #     cache.refresh_async(
    #         q.question, lambda text: call_model(retrieve_context(text), text)
    #     )
    #     # return {"answer": cached.decode()}
    #     return json.loads(cached.decode())

    # print("Getting answer from LLM")
    # context = retrieve_context(q.question)
    # # answer = call_model(context, q.question)
    # response = call_rag(context, q.question)
    # cache.set(q.question, json.dumps(response))
    # return response
    # semantic cache
    create_index()
    cached = search_similar_question(q.question)
    if cached:
        return cached

    print("Cache miss → calling RAG")
    context = retrieve_context(q.question)
    # print("context is", context)
    response = call_rag(context, q.question)
    save_to_cache(q.question, response)
    return response





if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
