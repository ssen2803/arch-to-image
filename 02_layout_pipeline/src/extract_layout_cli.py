"""הרצת המחלץ בלי מפתח API, דרך Claude Code במצב headless (`claude -p`).

אותו פרומפט מערכת, אותו מאמת ואותה לולאת תיקון כמו ב-extract_layout.py. ההבדלים:
  - אין structured outputs: הפלט הוא טקסט, וה-JSON נבדק מקומית (סכמה + גיאומטריה + ציטוטים).
  - המזהה שנשלח למודל ניטרלי (report_<hash>). שמות התיקיות מתארים את המקרה (למשל "underspecified_pit"),
    ושליחתם הייתה רמז לתשובה. אחרי החילוץ המזהה מוחלף בשם הדוגמה, כי השלבים הבאים משתמשים בו.
  - המודל רץ בלי כלים (`--tools ""`), בלי הגדרות משתמש ובתיקייה ריקה, ולכן אין לו גישה ל-gold.
  - כל ניסיון תיקון הוא קריאה חדשה שמכילה את הדו"ח, התשובה הקודמת ורשימת השגיאות.

שימוש:
    python src/extract_layout_cli.py examples/*/report.he.txt
    python src/extract_layout_cli.py examples/*/report.he.txt --model sonnet --out-name layout.pred_sonnet.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_layout import PROMPT_PATH, check, feedback_message, neutral_id, user_message  # noqa: E402
from layout_core import errors_only  # noqa: E402


def parse_json(text: str) -> dict:
    """מוציא את אובייקט ה-JSON מהתשובה, גם אם הוא עטוף ב-```json."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.index("{"): text.rindex("}") + 1]
    return json.loads(raw)


def call_claude(prompt: str, model: str, timeout: int = 600) -> tuple[str, dict]:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("הפקודה claude לא נמצאה")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
    with tempfile.TemporaryDirectory() as cwd:           # תיקייה ריקה: אין CLAUDE.md ואין קבצים
        cmd = [exe, "-p", "--model", model, "--tools", "", "--system-prompt-file", str(PROMPT_PATH),
               "--output-format", "json", "--no-session-persistence", "--setting-sources", "",
               "--strict-mcp-config"]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                           cwd=cwd, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude נכשל ({r.returncode}): {r.stderr[-500:] or r.stdout[-500:]}")
    data = json.loads(r.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude החזיר שגיאה: {data.get('result')}")
    return data["result"], data


def extract(report_text: str, source_id: str, model: str, attempts: int) -> tuple[dict, list, dict]:
    base = user_message(report_text, source_id)
    prompt, layout, issues = base, {}, []
    meta: dict = {"backend": "claude -p (headless, no tools)", "model": model, "attempts": []}
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        text, data = call_claude(prompt, model)
        entry = {"attempt": attempt, "seconds": round(time.perf_counter() - t0, 1),
                 "usage": {k: v for k, v in (data.get("usage") or {}).items() if isinstance(v, int)},
                 "models_used": sorted((data.get("modelUsage") or {}).keys())}
        try:
            layout = parse_json(text)
            issues = check(layout, report_text)
            errs = errors_only(issues)
            entry["errors"] = [str(e) for e in errs]
            feedback = feedback_message(errs) if errs else ""
        except (ValueError, KeyError) as exc:
            entry["errors"] = [f"JSON לא תקין: {exc}"]
            errs, feedback = [exc], f"Your answer was not a valid JSON object ({exc}). Return only the JSON object."
        meta["attempts"].append(entry)
        if not errs:
            break
        prompt = (f"{base}\n\nYour previous answer was:\n<previous>\n{text}\n</previous>\n\n{feedback}")
    meta["valid"] = not errors_only(issues) and bool(layout)
    return layout, issues, meta


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--model", default="opus", help="כינוי של Claude Code; המודל בפועל נשמר ב-meta")
    ap.add_argument("--out-name", default="layout.pred.json")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="הרצה מחדש גם אם יש layout.pred.json")
    args = ap.parse_args()
    for report in args.reports:
        out = report.parent / args.out_name
        if out.exists() and not args.force:
            print(f"{report.parent.name}: קיים, מדלג")
            continue
        text = report.read_text(encoding="utf-8")
        neutral = neutral_id(text)
        try:
            layout, issues, meta = extract(text, neutral, args.model, args.attempts)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"{report.parent.name}: נכשל - {exc}", flush=True)
            continue
        meta["source_id_sent"] = neutral
        layout.setdefault("source", {})["id"] = report.parent.name
        out.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report.parent.name}: {'תקין' if meta['valid'] else 'לא תקין'} אחרי "
              f"{len(meta['attempts'])} ניסיונות; אזהרות: {len(issues)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
