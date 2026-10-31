"""Build a consistent, base-corpus-only snapshot without changing local uploads."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import HybridSearchEngine

if __name__ == "__main__":
    target = ROOT / ".cloud-demo" / "snapshot"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "data" / "corpus.csv", target / "corpus.csv")
    engine = HybridSearchEngine(str(target / "corpus.csv"), index_dir=str(target / "index"))
    print(f"Prepared {len(engine.df)} documents in {target}")
