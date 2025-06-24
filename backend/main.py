import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from retriever import retrieve_context
from cache import Cache

import openai
import anthropic
import time
import random
import openai
from config import Config

USE_CLAUDE = os.getenv("USE_CLAUDE", "false").lower() == "true"
cache = Cache()
app = FastAPI()


class Query(BaseModel):
    question: str


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


@app.post("/ask")
async def ask(q: Query):
    cached = cache.get(q.question)
    if cached:
        cache.refresh_async(
            q.question, lambda text: call_model(retrieve_context(text), text)
        )
        return {"answer": cached.decode()}

    context = retrieve_context(q.question)
    answer = call_model(context, q.question)
    cache.set(q.question, answer)
    return {"answer": answer}
