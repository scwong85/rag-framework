import redis
import hashlib
from config import Config


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
