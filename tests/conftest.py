import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The evaluation/ scripts are top-level modules, not a package.
for extra in (ROOT / "src", ROOT / "evaluation"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
