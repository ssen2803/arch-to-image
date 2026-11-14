"""הגדרות משותפות לשלב 4: מחלקות, נתיבים, וחיבור לקוד של שלבים 2-3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
for p in (PROJECT / "02_layout_pipeline" / "src", PROJECT / "03_controlled_generation" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layout_core import CLASSES, PALETTE  # noqa: E402  ["background","floor","wall","installation","pit"]

NUM_CLASSES = len(CLASSES)          # 5
IGNORE = 255                        # פיקסל שלא נספר (אנשים וציוד בצילום אמיתי, מחוץ לשטח, לא ודאי)
CLASS_COLORS = [PALETTE[c] for c in CLASSES]

# הרזולוציה הגסה (כמו בשלב 3): בנוי / קרקע / בור
COARSE_NAMES = ["ground", "built", "pit"]
COARSE_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}   # background,floor->ground; wall,installation->built; pit->pit

DATA = ROOT / "data"
SYNTH_DIR = DATA / "synthetic"
REAL_DIR = DATA / "real"
RUNS = ROOT / "runs"
