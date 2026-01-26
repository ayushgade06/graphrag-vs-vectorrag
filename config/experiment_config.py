SEED = 42

# ===============================
# LLM CONFIGURATION
# ===============================
LLM_NAME = "Qwen/Qwen2-1.5B-Instruct"
LLM_MAX_TOKENS = 64

# ===============================
# EMBEDDING CONFIGURATION
# ===============================
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ===============================
# CHUNKING
# ===============================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ===============================
# RETRIEVAL
# ===============================
TOP_K = 10
CANDIDATE_POOL_SIZE = 200

# ===============================
# CONTEXT BUDGET
# ===============================
MAX_CONTEXT_TOKENS = 1024
MAX_CHUNKS = 8

# ===============================
# GRAPH SAFETY
# ===============================
MAX_GRAPH_DOCS = 1200

# ===============================
# DEVELOPMENT
# ===============================
DEV_MODE = False
SAMPLE_LIMIT = 5 if DEV_MODE else 10
