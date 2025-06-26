from openai import OpenAI
import numpy as np


class Embed:
    def __init__(self):
        self.openai_client = OpenAI()

    def embed(self, text):
        res = self.openai_client.embeddings.create(
            model="text-embedding-3-small", input=[text]
        )
        return np.array(res.data[0].embedding, dtype=np.float32)
