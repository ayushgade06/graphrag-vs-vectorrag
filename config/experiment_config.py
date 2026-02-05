SEED = 42

LLM_NAME = "Qwen/Qwen2-1.5B-Instruct"

USE_API_LLM = True

API_LLM_NAME = "nvidia/llama-3.1-8b-instruct"

LLM_MAX_TOKENS = 768

EMBEDDING_MODEL = "models/BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

TOP_K = 15
CANDIDATE_POOL_SIZE = 200

MAX_CONTEXT_TOKENS = 2048
MAX_CHUNKS = 8

MAX_GRAPH_DOCS = 80

DEV_MODE = False
SAMPLE_LIMIT = 5 if DEV_MODE else 10
