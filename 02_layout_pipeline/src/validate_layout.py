"""אימות קובצי Layout מול הסכמה, מול הכללים הגיאומטריים, ומול טקסט המקור.

שימוש:
    python src/validate_layout.py examples/*/layout.gold.json
    python src/validate_layout.py my_layout.json --text my_report.txt --tol 0.5

אם לא ניתן --text, מחפשים report*.txt באותה תיקייה; אם נמצא - נבדקת גם
עקיבות הציטוטים (source_span) מול הטקסט.
קוד יציאה 0 אם אין שגיאות (אזהרות מותרות), 1 אחרת.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_core import (  # noqa: E402
    DEFAULT_TOL_M, check_spans, errors_only, find_report_text, load_layout, validate,
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="אימות קובצי ArchLayout")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_M, help="סובלנות גיאומטרית במטרים")
    ap.add_argument("--text", type=Path, help="טקסט המקור לבדיקת עקיבות הציטוטים")
    args = ap.parse_args()

    failed = 0
    for path in args.paths:
        layout = load_layout(path)
        issues = validate(layout, tol=args.tol)
        text_path = args.text or find_report_text(path)
        if text_path and not errors_only(issues):
            issues += check_spans(layout, text_path.read_text(encoding="utf-8"))
        errs = errors_only(issues)
        status = "תקין" if not errs else "לא תקין"
        print(f"{path}: {status} ({len(errs)} שגיאות, {len(issues) - len(errs)} אזהרות)")
        for issue in issues:
            print(f"    {issue}")
        failed += bool(errs)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
