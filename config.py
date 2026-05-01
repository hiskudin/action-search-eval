from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SERVER_URL = "http://localhost:5117"
NUM_DAYS = 10
EMBED_DIM = 384
