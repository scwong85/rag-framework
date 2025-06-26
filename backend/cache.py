import redis
import hashlib
from config import Config
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from openai import OpenAI
import json


class Cache:
    def __init__(self):
        self.client = redis.from_url(Config.REDIS_URL)

    def _key(self, text):
        return "cache:" + hashlib.md5(text.encode()).hexdigest()

    def get(self, text):
        key = self._key(text)
        result = self.client.get(key)
        return result

    def set(self, text, result, ttl=86400):
        key = self._key(text)
        pipe = self.client.pipeline()
        pipe.set(key, result, ex=ttl)
        pipe.execute()

    def refresh_async(self, text, compute_func):
        from threading import Thread

        def refresh():
            result = compute_func(text)
            self.set(text, result)

        Thread(target=refresh).start()

    def create_index(self, VECTOR_INDEX, EMBEDDING_DIM):
        try:
            self.client.ft(VECTOR_INDEX).info()
        except:
            print("Index not found, creating...")
            self.client.ft(VECTOR_INDEX).create_index(
                fields=[
                    TextField("question"),
                    TextField("answer"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": EMBEDDING_DIM,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                ],
                definition=IndexDefinition(prefix=["q:"], index_type=IndexType.HASH),
            )

    def embed(self, text):
        openai_client = OpenAI()
        res = openai_client.embeddings.create(
            model="text-embedding-3-small", input=[text]
        )
        return np.array(res.data[0].embedding, dtype=np.float32)

    def save_to_cache(self, question: str, answer: dict, VECTOR_INDEX: str):
        vector = self.embed(question).tobytes()

        self.client.ft(VECTOR_INDEX).add_document(
            f"q:{question}",
            replace=True,
            question=question,
            answer=json.dumps(answer),
            embedding=vector,
        )

    def search_similar_question(self, question, VECTOR_INDEX, threshold=0.5):
        print("searching for answer for question", question)
        vector = self.embed(question).tobytes()
        q = (
            Query("*=>[KNN 1 @embedding $vec_param AS score]")
            .sort_by("score")
            .return_fields("question", "answer", "score")
            .dialect(2)
        )
        params = {"vec_param": vector}

        res = self.client.ft(VECTOR_INDEX).search(q, query_params=params)
        if res.total > 0 and float(res.docs[0].score) < threshold:
            print("cached matched")
            return json.loads(res.docs[0].answer)
        return None
