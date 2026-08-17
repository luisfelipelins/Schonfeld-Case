from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR   = ROOT_DIR / "data"
DATA_ZIP   = DATA_DIR / "zip"
DATA_RAW   = DATA_DIR / "raw"
DATA_INT   = DATA_DIR / "int"
DATA_FINAL = DATA_DIR / "final"
DATA_FACTORS = DATA_DIR / "factors"
START_REF = "2014-01-01"

for folder in [DATA_ZIP, DATA_RAW, DATA_INT, DATA_FINAL, DATA_FACTORS]:
    folder.mkdir(parents=True, exist_ok=True)
