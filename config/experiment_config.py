SEED = 42 

# llm config
LLM_NAME = "Qwen/Qwen2-1.5B-Instruct"
LLM_MAX_TOKENS = 128

# embedding config
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# retrieval
TOP_K = 20
CANDIDATE_POOL_SIZE = 200

# context budget of llm
MAX_CONTEXT_TOKENS = 1024
MAX_CHUNKS = 12

# system safety in case of graphrag
MAX_GRAPH_DOCS = 1200

# developement mode
DEV_MODE = False
SAMPLE_LIMIT = 5 if DEV_MODE else 10
