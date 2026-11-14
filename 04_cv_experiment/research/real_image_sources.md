# Real top-down excavation imagery: sources for the test set

Checked 2026-11-14. I opened each record, API or page myself unless it is marked unverified.

## Candidate sources

| # | Source | Contents | Near-nadir | License | Labels | Suit. |
|---|---|---|---|---|---|---|
| 1 | Uziel, Zohar & Jaffe, JAS 185:106430 (2026). Data and code: https://doi.org/10.5281/zenodo.15673668 | 9 Israeli drone orthomosaics (Tel Gezer, Beit She'arim, Rosh Zayit, Kh. Qana ×2, Kh. Dorban, Kh. Garas, Kh. Sum, Yodfat; 11.5 GB). Also 44 stone tiles and 56 wall tiles (800×800 px, LabelMe polygons), plus models | 100 labeled tiles; unlimited crops | MIT | Yes: stones and walls only | 5 |
| 2 | Commons `Category:Aerial photographs of archaeological sites in Israel` (https://commons.wikimedia.org/wiki/Category:Aerial_photographs_of_archaeological_sites_in_Israel) | 696 files across 60 subcategories. 261 are drone series by user Bukvoed (e.g. `Khirbet-Qeiyafa-V3-*`) | est. 100–200 | Per file: CC BY 4.0 (262), CC BY-SA 4.0 (214), CC BY-SA 3.0 (112), CC BY 2.5 (63), PD (32) | No | 4–5 |
| 3 | Zenodo: CaNova (https://doi.org/10.5281/zenodo.23492096); Berenike graves (https://zenodo.org/records/13937061); Berenike trenches (https://zenodo.org/records/18904652) | Drone capture and photogrammetry; orthophoto + DTM; trench documentation (3–13 GB each) | Crops; archives not opened | CC BY-SA 4.0 / CC BY 4.0 | No | 3 |
| 4 | Open Context: Kenan Tepe (https://opencontext.org/projects/3de4cd9c-259e-4c14-9b03-8b10454ba66e); Tel Dor Area G (https://opencontext.org/projects/b028c160-092e-4e2d-9738-a40367ad543b) | Trench photos captioned "Overhead shot"; Tel Dor report figures, mostly oblique | ~48; <10 | CC BY 4.0 | No | 3 / 2 |
| 5 | Sketchfab, downloadable models matching "excavation" | Trench photogrammetry models; top-down views must be rendered | ≥360 CC BY, 308 BY-SA, 108 CC0 | Per model | No | 3 |
| 6 | ADS (https://archaeologydataservice.ac.uk/about/policies/use-access-to-data/); tDAR (https://tdar.org) | UK and US archives; tDAR needs a login | Not counted | ADS: per collection (CC BY 4.0 recommended). tDAR: CC BY 3.0 default (unverified) | No | 2 |
| 7 | HA-ESI (https://hadashot.iaa.org.il) | Excavation reports with photos and plans | Many | © IAA | No | 1 |
| 8 | Schnorr & Leimkühler, JCAA (2026) (https://doi.org/10.5281/zenodo.21507975) | 5,560 stone masks on vertical wall-face orthomosaics | 0 (wrong viewpoint) | CC BY 4.0 | Yes | 2 |

These had no usable data:
- **Structura:** code only.
- **Altaweel et al. (2022):** no public dataset found.
- **Hugging Face, Kaggle, Roboflow:** no labeled excavation dataset found.
- **Israeli Sketchfab models** (Tell Keisan, Megiddo, Timna): not downloadable.

## IAA / HA-ESI terms
- **HA-ESI footer:** "Figures, websites, texts and photos © Israel Antiquities Authority".
- **HA-ESI About page:** "Prints of reports are available from the web site for personal and educational use only."
- **HA-ESI author guidelines:** "Please note that photographs are protected by copyright laws."
- **IAA terms of use** (https://www.iaa.org.il/page/usage, Hebrew; my translation):
  - Copying, modifying, distributing or "ליצור עבודות הנגזרות" (creating derivative works) requires prior written consent.
  - Fair use is preserved.
  - Systematic copying and download bots are "absolutely prohibited".
  - Use is for personal and private purposes only.
- **Photo archive:** the IAA's "ארכיון צילומים" link redirects to the homepage, and iaa-archives.org.il did not resolve. The only openly licensed IAA images I found are 116 Mandate-era photos on Commons, under CC BY-SA 4.0 by permission (mostly oblique).

Do not use HA-ESI images without written permission from the IAA.

## Licensing pitfalls
- **No-derivatives (ND):** CC BY-ND 4.0 allows you to "produce and reproduce, but not Share, Adapted Material". You can annotate privately but cannot publish the masks. Exclude ND images.
- **Non-commercial (NC):** fine for coursework, but it blocks any later commercial release.
- **ShareAlike (SA):** if your masks count as adaptations, you must release them under the same license.
- **Attribution:** give the author, URL, license and what you changed. MIT requires keeping its notice, and Uziel et al. also ask you to cite their paper.
- **Licenses vary per file or collection:** keep a manifest CSV (url, author, license).
- **Zenodo mirrors:** records labelled "Objaverse 1.0 / Sketchfab" are third-party copies. Check the license on the original page.

## Recommendation
1. **Uziel et al. (row 1):** ~25 crops from Tel Gezer, Beit She'arim, Rosh Zayit and Kh. Qana. Start from their wall polygons and add the other classes. The sites are mostly unexcavated, so expect few floors and installations.
2. **Bukvoed's CC BY 4.0 drone shots (row 2):** ~30–40 near-nadir images of excavated sites such as Qeiyafa, Lakhish and Wadi Hamam.
3. **Rows 3–4:** ~10–15 images for pits, graves and hearths.

## Not verified
- The near-nadir counts are estimates. For Commons I sampled 2 gallery pages and tallied authors; I did not download the Zenodo archives; for Open Context I only searched captions.
- The ground resolution of the Uziel orthomosaics and how much of each site is excavated.
- tDAR's terms page returned HTTP 403.
- IAA Visual Archive terms and fees, and whether the iaa.org.il terms also govern HA-ESI.
- Data access for the HIDA/IDSI Negev terrace/wall challenge.
- How many suitable images ADS holds. I did not search Papers with Code.
