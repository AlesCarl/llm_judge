import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# config.py lives at <repo>/src/nika/config.py → repo root is two levels up
BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)
RESULTS_DIR = os.getenv("RESULTS_DIR") or f"{BASE_DIR}/results"
