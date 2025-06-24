import os


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    USE_CLAUDE = os.getenv("USE_CLAUDE", "false").lower() == "true"
    MAX_RETRIES = 3
