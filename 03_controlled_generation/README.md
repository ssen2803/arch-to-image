# שלב 3: יצירה מבוקרת (Layout ← תמונה), מותאם ל-6GB VRAM

השלב הזה מקבל מסכת סגמנטציה, או Layout משלב 2 שממנו המסכה נגזרת, ומייצר תמונה ב-512×512. הייצור מתבצע ב-Stable Diffusion 1.5 יחד עם ControlNet-seg. השלב כולל גם את שיטת המדידה להזיות אובייקטים, ואת הכיול שלה על תמונות אמיתיות.

```
layout.json ─► מסכת 5 מחלקות ─► מסכה בצבעי ADE20K ─► SD1.5 + ControlNet-seg ─► תמונה 512×512
  (שלב 2)        (פלטת פרויקט)    (config/class_mapping)   (FP16 · xformers · offload)
                                                                        │
מדידה: SegFormer-ADE20K ─► מפת תוויות ─► HOR / HOR_area / Recall / HAR ◄┘   (measurement_HOR.md)
```

## אילוצי החומרה בקוד, ומה נמדד בפועל

נמדד על **RTX 3050 Laptop 6GB**: 80 תמונות, 30 צעדי UniPC.

| אילוץ | המימוש ([src/generate.py](src/generate.py), `build_pipeline`) | הערה |
|---|---|---|
| **FP16** | `dtype=torch.float16`, `variant="fp16"` | קובצי המשקולות שמורדים הם כבר בחצי דיוק |
| **xformers** | `pipe.enable_xformers_memory_efficient_attention()` | אם אין wheel תואם, נעשה מעבר מבוקר ל-SDPA של PyTorch, והמעבר מתועד ב-meta. ברצה הזו xformers 0.0.35 פעל |
| **CPU offloading** | `pipe.enable_model_cpu_offload()` | כל תת-מודל עולה ל-GPU רק בזמן השימוש בו. האפשרות `--offload sequential` חוסכת עוד, אבל איטית פי כמה |
| **זיכרון שיא** | **3,111–3,119MB** מתוך 6,144MB (כ-51%) | נמדד ב-`torch.cuda.max_memory_allocated` |
| **זמן** | **9.4 שניות לתמונה** בממוצע | |

**למה צריך מיפוי ל-ADE20K:** ControlNet-seg אומן על צבעי ADE20K, ולכן אי אפשר להזין לו את פלטת הפרויקט כפי שהיא. כל מחלקה שלנו נצבעת בצבע של מחלקת ADE20K שמשמעותה קרובה:

| מחלקה שלנו | מחלקת ADE20K |
|---|---|
| רקע | earth |
| רצפה | floor |
| קיר | wall |
| מתקן | fireplace |
| בור | swimming pool |

המיפוי נמצא ב-[config/class_mapping.json](config/class_mapping.json). הפלטה המלאה של ADE20K נלקחה מקובץ המקור של ControlNet ונשמרה ב-[data/ade20k_palette.json](data/ade20k_palette.json). אפשר לדרוס את המיפוי משורת הפקודה, למשל `--map pit=sand`.

## פקודות

ההתקנה מתבצעת פעם אחת, משורש הפרויקט. הפקודות המלאות נמצאות בראש [requirements.txt](requirements.txt).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r 03_controlled_generation\requirements.txt
```

מתוך התיקייה `03_controlled_generation`, כאשר `PY` מפנה ל-`..\.venv\Scripts\python`:

```bash
# ייצור
PY src/generate.py --layout ../02_layout_pipeline/examples/ex01_iron_age_room/layout.gold.json
PY src/generate.py --layout ../02_layout_pipeline/examples/*/layout.gold.json --style photo --seeds 0 1 2 3
PY src/generate.py --mask my_seg.png --prompt "top-down plan of a stone house"
PY src/generate.py --layout ... --cond-scale 0          # תנאי בסיס text-only (אותו מודל ו-seed, בלי מידע מרחבי)
PY src/generate.py --layout ... --map pit=sand --tag pitsand   # ניסוי A/B על המיפוי

# מדידה
PY src/evaluate.py --layout <layout.json> --images outputs/<name>/photo_s*.png
PY src/summarize.py                                     # כל התמונות, כל התנאים -> outputs/summary.md

# בדיקות (לא דורשות GPU)
PY -m pytest tests -q
```

## מה יש ב-outputs

לכל דוגמה, `outputs/<example>/` מכילה:
- `control_ade20k.png`: מסכת ההתניה שהוזנה ל-ControlNet.
- `<condition>_s<seed>.png`: התמונה שנוצרה.
- `<condition>_meta.json`: הפרומפט, הפרמטרים, זיכרון השיא והזמן של כל seed.
- `*.hor.json`: המדדים בשתי הרזולוציות, יחד עם רשימת ההזיות והאלמנטים החסרים.
- `*.hor.png`: המחשה. אדום מסמן הזיה, וצהוב מקווקו מסמן אלמנט חסר.

הסיכום הכולל נמצא ב-`outputs/summary.md` וב-`outputs/summary.csv`.

חמישה תנאים, כל אחד על 5 דוגמאות עם 4 seeds:

| תנאי | מה הוא |
|---|---|
| `schematic` | הסגנון הראשי לפי ההכרעה משלב 1 |
| `photo` | סגנון צילום, להשוואה, וגם הסגנון שבו המעריך עבר כיול |
| `photo_textonly` | תנאי הבסיס, בלי התניה מרחבית |
| `photo_bgsand` | ניסוי A/B שבו הרקע ממופה ל-sand |
| `gligen` | GLIGEN עם תיבה וביטוי טקסט לכל אלמנט נראה, במקום מסכה (#14) |

## תוצאות עיקריות

ההגדרה המלאה, הכיול והיסטוריית התיקונים נמצאים ב-[measurement_HOR.md](measurement_HOR.md).

| (סגנון צילום, רזולוציה `coarse`, 5 Layouts × 4 seeds) | HOR ↓ | HOR_area ↓ | Recall ↑ | HAR ↓ |
|---|---|---|---|---|
| ControlNet | **0.625** | **0.729** | **0.632** | 0.640 |
| text-only | 0.823 | 0.775 | 0.203 | **0.219** |
| GLIGEN (תיבה וביטוי לכל אלמנט, #14) | 0.763 | 0.787 | 0.497 | 0.569 |

1. **ControlNet מצייר את ה-Layout.** ה-Recall עולה פי שלושה, ושיעור ההזיה יורד. ControlNet טוב יותר ב-17 מתוך 20 זוגות מזווגים בשני המדדים.
2. **ControlNet "סוגר" חדרים.** בשטח שאמור להיות ריק הוא מצייר קירות ושורות אבנים מומצאים. HAR גבוה פי שלושה, ו-ControlNet טוב יותר רק ב-3 מתוך 20 זוגות. ההסבר הסביר הוא הנטייה שנלמדה מ-ADE20K, שבו רצפה כמעט תמיד מוקפת קירות. זו ההזיה העיקרית שצריך לטפל בה.
3. **טקסט חסר משמעות מופיע בפינות** גם בסגנון הסכמטי וגם בסגנון הצילום, למרות שה-negative prompt אוסר טקסט. המדד סופר אותו כאובייקט זר.
4. **הסגנון הסכמטי נאמן מאוד ל-Layout בעין,** למשל הפתח בקיר המחיצה ב-ex04. אבל עם המעריך הנוכחי **אי אפשר למדוד אותו**: SegFormer רואה באיור פוסטר או מסך.
5. **קידוד הראיות לא עובר דרך ControlNet.** קיר היפותטי מצויר כמו קיר שנחשף, כי מסכת ADE20K לא נושאת מידע על רמת ודאות.
6. **החלפת הרקע ל-sand החמירה את HAR** (0.772 מול 0.640), ולכן earth נשאר ברירת המחדל. ההחלטה התקבלה על פי המדד.
7. **GLIGEN אינו תחליף ל-ControlNet** (ניסוי #14, [src/generate_gligen.py](src/generate_gligen.py), 20 תמונות באותו פרוטוקול, 15 שניות ו-2.8GB לתמונה).
   - **מול text-only:** הוא טוב יותר ב-14 מתוך 20 זוגות ב-HOR וב-Recall, כלומר התיבות כן מעבירות מידע מרחבי.
   - **מול ControlNet:** הוא טוב יותר רק ב-4 מתוך 20 זוגות ב-HOR וב-2 מתוך 20 ב-Recall. מנגד, הוא ממציא פחות בשטח הריק: טוב יותר ב-14 מתוך 20 ב-HAR.
   - **מראה:** התמונות נוטות למראה של פוסטר, עם טקסט חסר משמעות (רכיבים זרים: 3.85 מול 2.10).
   - **השאלה שבגללה נבדק לא נענתה:** האם ביטוי כמו "tabun oven" גורם לטאבון להיראות כטאבון. SegFormer לא מכיר טאבון, ולכן המעריך הנוכחי לא יכול למדוד את זה. צריך מעריך ברמת התיבה, למשל CLIP או VQA על כל אזור.
   - **מגבלות:** GLIGEN מבוסס על SD 1.4 ולא על 1.5, ו-diffusers הוציאו אותו משימוש. הכיוון שנשאר פתוח הוא ControlNet לגאומטריה, יחד עם ביטויים אזוריים בפרומפט.
8. **מסנן הבטיחות של SD מחזיר תמונה שחורה** גם על תמונות חפירה תמימות: 1 מתוך 20 ב-GLIGEN, ו-2 מתוך 600 בנתונים הסינתטיים. בכל המחוללים יש עכשיו ניסיון חוזר עם seed + 1000 (`run_with_retry`), וה-seed שבו השתמשו בפועל נשמר ב-meta. בבדיקה של 80 התמונות הקודמות לא נמצאה אף תמונה שחורה.

## השלבים הבאים, לפי הממצאים

| ממצא | צעד מוצע |
|---|---|
| סגנון סכמטי לא מדיד (ממצא 4), וקידוד ראיות שלא עובר (ממצא 5) | אימון LoRA על SD1.5 עם זוגות של מסכה ורינדור סכמטי בפלטת הפרויקט. את הזוגות ייצר `render_layout.py` משלב 2 מ-Layouts סינתטיים. אחרי האימון `PaletteEvaluator` תקף, ואפשר לקודד בו גם רמות ודאות |
| "סגירת חדרים" (ממצא 2) | ה-LoRA צפוי להחליש את הנטייה שנלמדה מ-ADE20K. אפשרות נוספת היא להוריד את `--cond-scale` בשטח הרקע |
| בורות כמעט לא מזוהים (Recall של 0.20 ב-ex02) | ניסוי A/B על המיפוי `pit=sand` או `pit="dirt track"`, באותו פרוטוקול |
| HOR מוחלט של כ-0.6 כולל רעש של המעריך | ביקורת אנושית על 30 תמונות (measurement_HOR.md, סעיף 6.4), לפני כל דיווח סופי |

## בדיקות

52 בדיקות ב-[tests/test_hallucination.py](tests/test_hallucination.py), ואף אחת מהן לא דורשת GPU. הן מכסות:
- מקרים סינתטיים עם תשובה ידועה: אובייקט נוסף, מחלקה שגויה, תזוזה בתוך הסובלנות, אובייקט זר, תמונה ריקה ורעש.
- שלושת הכשלים שהתגלו בכיול: הזיה שמחוברת למבנה אמיתי, פיצול של הזיה לכמה רכיבים, ושטח שמחוץ לגבול החפירה.
- בדיקת שפיות של כל השרשרת על חמש הדוגמאות.
- עקביות המיפוי ל-ADE20K.
