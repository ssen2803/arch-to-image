# סיכום הזיות (מעריך: nvidia/segformer-b4-finetuned-ade-512-512, 80 תמונות)

> פיילוט: 5 Layouts. המספרים תיאוריים, ולא מבחן מובהקות. בסגנון schematic, SegFormer מפרש את האיור כפוסטר או מסך (כשל תחום שזוהה בכיול), ולכן המספרים שם אינם תקפים.

## רזולוציה: fine

| תנאי | HOR ↓ | HOR_area ↓ | Object Recall ↑ | HAR ↓ | רכיבים זרים ↓ |
|---|---|---|---|---|---|
| photo | 0.732 | 0.784 | 0.321 | 0.678 | 2.100 |
| photo_bgsand | 0.734 | 0.825 | 0.296 | 0.784 | 1.650 |
| photo_textonly | 0.839 | 0.795 | 0.116 | 0.213 | 0.650 |
| schematic ⚠ לא תקף | 0.779 | 0.845 | 0.311 | 0.999 | 6.100 |

השוואה מזווגת: **photo** מול **photo_textonly** (אותו Layout ואותו seed, 20 זוגות). מספר הזוגות שבהם photo טוב יותר:

- HOR: 14 מתוך 20 (שוויון: 3)
- HOR_area: 10 מתוך 20 (שוויון: 3)
- object_recall: 12 מתוך 20 (שוויון: 8)
- HAR: 3 מתוך 20

השוואה מזווגת: **photo_bgsand** מול **photo_textonly** (אותו Layout ואותו seed, 20 זוגות). מספר הזוגות שבהם photo_bgsand טוב יותר:

- HOR: 13 מתוך 20 (שוויון: 4)
- HOR_area: 10 מתוך 20 (שוויון: 2)
- object_recall: 11 מתוך 20 (שוויון: 9)
- HAR: 1 מתוך 20

השוואה מזווגת: **photo_bgsand** מול **photo** (אותו Layout ואותו seed, 20 זוגות). מספר הזוגות שבהם photo_bgsand טוב יותר:

- HOR: 7 מתוך 20 (שוויון: 7)
- HOR_area: 9 מתוך 20 (שוויון: 5)
- object_recall: 0 מתוך 20 (שוויון: 18)
- HAR: 4 מתוך 20 (שוויון: 4)

לפי Layout (HOR_area / recall, ממוצע על seeds):

| Layout | photo | photo_bgsand | photo_textonly | schematic |
|---|---|---|---|---|
| ex01_iron_age_room | 0.58 / 0.44 | 0.72 / 0.38 | 0.88 / 0.00 | 0.82 / 0.44 |
| ex02_chalcolithic_pits | 1.00 / 0.00 | 1.00 / 0.00 | 0.75 / 0.00 | 1.00 / 0.00 |
| ex03_byzantine_winepress | 0.93 / 0.20 | 0.93 / 0.20 | 0.94 / 0.05 | 0.95 / 0.15 |
| ex04_hellenistic_two_rooms | 0.73 / 0.47 | 0.77 / 0.41 | 0.73 / 0.16 | 0.72 / 0.47 |
| ex05_roman_courtyard | 0.69 / 0.50 | 0.71 / 0.50 | 0.68 / 0.38 | 0.74 / 0.50 |

## רזולוציה: coarse

| תנאי | HOR ↓ | HOR_area ↓ | Object Recall ↑ | HAR ↓ | רכיבים זרים ↓ |
|---|---|---|---|---|---|
| photo | 0.625 | 0.729 | 0.632 | 0.640 | 2.100 |
| photo_bgsand | 0.593 | 0.771 | 0.642 | 0.772 | 1.650 |
| photo_textonly | 0.823 | 0.775 | 0.203 | 0.219 | 0.650 |
| schematic ⚠ לא תקף | 0.706 | 0.814 | 0.578 | 0.984 | 6.100 |

השוואה מזווגת: **photo** מול **photo_textonly** (אותו Layout ואותו seed, 20 זוגות). מספר הזוגות שבהם photo טוב יותר:

- HOR: 17 מתוך 20
- HOR_area: 15 מתוך 20
- object_recall: 17 מתוך 20 (שוויון: 3)
- HAR: 3 מתוך 20

השוואה מזווגת: **photo_bgsand** מול **photo_textonly** (אותו Layout ואותו seed, 20 זוגות). מספר הזוגות שבהם photo_bgsand טוב יותר:

- HOR: 17 מתוך 20
- HOR_area: 14 מתוך 20
- object_recall: 17 מתוך 20 (שוויון: 3)
- HAR: 1 מתוך 20

השוואה מזווגת: **photo_bgsand** מול **photo** (אותו Layout ואותו seed, 20 זוגות). מספר הזוגות שבהם photo_bgsand טוב יותר:

- HOR: 8 מתוך 20 (שוויון: 5)
- HOR_area: 8 מתוך 20 (שוויון: 2)
- object_recall: 3 מתוך 20 (שוויון: 15)
- HAR: 4 מתוך 20 (שוויון: 3)

לפי Layout (HOR_area / recall, ממוצע על seeds):

| Layout | photo | photo_bgsand | photo_textonly | schematic |
|---|---|---|---|---|
| ex01_iron_age_room | 0.52 / 0.75 | 0.64 / 0.83 | 0.87 / 0.00 | 0.79 / 0.83 |
| ex02_chalcolithic_pits | 0.91 / 0.20 | 0.95 / 0.20 | 0.74 / 0.00 | 0.98 / 0.10 |
| ex03_byzantine_winepress | 0.84 / 0.50 | 0.84 / 0.50 | 0.90 / 0.12 | 0.88 / 0.38 |
| ex04_hellenistic_two_rooms | 0.70 / 0.96 | 0.73 / 0.88 | 0.72 / 0.29 | 0.69 / 0.83 |
| ex05_roman_courtyard | 0.67 / 0.75 | 0.69 / 0.80 | 0.65 / 0.60 | 0.73 / 0.75 |

