# config/paths.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Memory
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory_store")
MEMORY_INDEX_PATH = os.path.join(MEMORY_DIR, "mem_index.faiss")
MEMORY_TEXTS_PATH = os.path.join(MEMORY_DIR, "memories.json")
MEMORY_REINDEX_EVERY = max(1, int(os.getenv("MEMORY_REINDEX_EVERY", "1")))

# Syllabus vector store
INDEX_DIR = os.path.join(PROJECT_ROOT, "vector_store")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")
INGEST_MANIFEST_PATH = os.path.join(INDEX_DIR, "ingest_manifest.json")
INGEST_CACHE_DIR = os.path.join(INDEX_DIR, "cache")

# Course catalog
COURSES_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "courses.json")

# App behavior
RESET_MEMORY_ON_INDEX = os.getenv("RESET_MEMORY_ON_INDEX", "1") == "1"
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.60"))
ENABLE_REQUEST_LOG = os.getenv("ENABLE_REQUEST_LOG", "1") == "1"

# Models
E5_MODEL_NAME = os.getenv("E5_MODEL_NAME", "intfloat/multilingual-e5-large")
E5_LOCAL_ONLY = os.getenv("E5_LOCAL_ONLY", "1") == "1"
QWEN_API_URL = os.getenv("QWEN_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507-FP8")
QWEN_MAX_TOKENS = 2048
