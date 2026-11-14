"""המרת תיאור חפירה (טקסט) ל-Layout מרחבי בעזרת Claude.

זרימה:
  1. שליחת הטקסט + פרומפט המערכת (prompts/text_to_layout_system.md) ל-Claude,
     עם structured outputs - התשובה מובטחת כ-JSON תואם לגרסת ה-API של הסכמה.
  2. אימות מקומי מלא: סכמה קפדנית + גיאומטריה + עקיבות ציטוטים.
  3. אם יש שגיאות - מחזירים אותן למודל כמשוב ומבקשים תיקון (עד --attempts).
  4. שמירת layout.pred.json + קובץ meta עם תיעוד התהליך (paradata).

שימוש:
    python src/extract_layout.py examples/ex01_iron_age_room/report.he.txt
    python src/extract_layout.py examples/*/report.he.txt --attempts 3
    python src/extract_layout.py my_report.txt --out my_layout.json
    python src/extract_layout.py my_report.txt --dry-run   # מדפיס את הבקשה בלי לקרוא ל-API

הרשאות: ANTHROPIC_API_KEY, או פרופיל שנוצר ב-`ant auth login`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_core import ROOT, Issue, api_schema, check_spans, errors_only, validate  # noqa: E402

MODEL = "claude-opus-5-5"
PROMPT_PATH = ROOT / "prompts" / "text_to_layout_system.md"
# fallback בצד השרת: אם מסווג בטיחות דוחה את הבקשה, השרת מריץ אותה מחדש על מודל חלופי.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def user_message(report_text: str, source_id: str) -> str:
    return f"source.id: {source_id}\n\n<report>\n{report_text.strip()}\n</report>"


def feedback_message(errors: list[Issue]) -> str:
    lines = "\n".join(f"- {e.where}: {e.message}" for e in errors)
    return ("The layout failed validation. Fix every error below and return the full corrected JSON. "
            "Keep elements and relations that were already correct.\n" + lines)


def build_request(messages: list, system_prompt: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 64000,
        "system": system_prompt,
        "messages": messages,
        "output_config": {
            "effort": "high",
            "format": {"type": "json_schema", "schema": api_schema()},
        },
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def check(layout: dict, report_text: str) -> list[Issue]:
    issues = validate(layout)
    if not errors_only(issues):
        issues += check_spans(layout, report_text)
    return issues


def extract(report_text: str, source_id: str, attempts: int = 3) -> tuple[dict, list[Issue], dict]:
    import anthropic  # ייבוא מאוחר: --dry-run ובדיקות לא דורשים את ה-SDK

    client = anthropic.Anthropic()
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    messages: list = [{"role": "user", "content": user_message(report_text, source_id)}]
    meta = {"model": MODEL, "attempts": [], "served_by": None}
    layout, issues = {}, []

    for attempt in range(1, attempts + 1):
        with client.beta.messages.stream(**build_request(messages, system_prompt)) as stream:
            response = stream.get_final_message()
        meta["served_by"] = response.model

        if response.stop_reason == "refusal":
            raise RuntimeError(f"הבקשה נדחתה ע\"י המודל: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("התשובה נקטעה (max_tokens). קצר את הטקסט או פצל אותו.")

        text = next(b.text for b in response.content if b.type == "text")
        layout = json.loads(text)
        issues = check(layout, report_text)
        errs = errors_only(issues)
        meta["attempts"].append({"attempt": attempt, "errors": [str(e) for e in errs],
                                 "request_id": response._request_id})
        if not errs:
            break
        # שומרים את תוכן התשובה המלא (כולל בלוקי thinking) ומוסיפים משוב לתיקון
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": feedback_message(errs)})

    meta["valid"] = not errors_only(issues)
    return layout, issues, meta


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="טקסט חפירה -> ArchLayout JSON בעזרת Claude")
    ap.add_argument("reports", nargs="+", type=Path, help="קובצי טקסט של תיאורי חפירה")
    ap.add_argument("--out", type=Path, help="נתיב פלט (רק כשיש קובץ קלט יחיד)")
    ap.add_argument("--attempts", type=int, default=3, help="מספר ניסיונות תיקון מקסימלי")
    ap.add_argument("--dry-run", action="store_true", help="הדפסת הבקשה בלבד, ללא קריאה ל-API")
    args = ap.parse_args()
    if args.out and len(args.reports) > 1:
        ap.error("--out אפשרי רק עם קובץ קלט יחיד")
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print("חסר ה-SDK של Anthropic: pip install -r requirements.txt")
            return 1

    for report in args.reports:
        text = report.read_text(encoding="utf-8")
        source_id = report.parent.name if report.parent != Path(".") else report.stem
        out = args.out or report.parent / "layout.pred.json"

        if args.dry_run:
            req = build_request([{"role": "user", "content": user_message(text, source_id)}],
                                PROMPT_PATH.read_text(encoding="utf-8"))
            req["system"] = f"<{len(req['system'])} chars from {PROMPT_PATH.name}>"
            req["output_config"]["format"]["schema"] = f"<api_schema(): {len(json.dumps(api_schema()))} chars>"
            print(json.dumps(req, ensure_ascii=False, indent=2))
            continue

        # ה-SDK כבר מבצע retry אוטומטי ל-429/5xx ולשגיאות רשת; כאן רק מדווחים בצורה קריאה.
        try:
            layout, issues, meta = extract(text, source_id, args.attempts)
        except anthropic.AuthenticationError:
            print("אין הרשאה ל-API: הגדר ANTHROPIC_API_KEY או הרץ `ant auth login`.")
            return 1
        except anthropic.RateLimitError:
            print("חריגה ממגבלת קצב גם אחרי ניסיונות חוזרים. נסה שוב מאוחר יותר.")
            return 1
        except anthropic.APIStatusError as exc:
            print(f"{report}: שגיאת API {exc.status_code}: {exc.message}")
            return 1
        except anthropic.APIConnectionError:
            print("שגיאת רשת: אין חיבור ל-API.")
            return 1
        except (RuntimeError, json.JSONDecodeError) as exc:
            print(f"{report}: נכשל - {exc}")
            return 1

        out.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "תקין" if meta["valid"] else "לא תקין"
        print(f"{report} -> {out}: {status} אחרי {len(meta['attempts'])} ניסיונות")
        for issue in issues:
            print(f"    {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
