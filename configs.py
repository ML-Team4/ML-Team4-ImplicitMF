import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "datasets"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
RESULT_DIR = ROOT_DIR / "results"
BASELINE_MODEL_DIR = ROOT_DIR / "checkpoints/grid_best_implicit_mf.npz"


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

GAMMA_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

RELEASE_URL = "https://github.com/<아이디>/<저장소명>/releases/download/<태그명>"