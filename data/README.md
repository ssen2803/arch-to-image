# Data

Four data sets. Only small samples are in git; the rest is generated, released or downloaded as described here.

| Set | Size | Source | Licence | Where |
|---|---|---|---|---|
| **Example reports** | 8 Hebrew texts, 40 elements, 52 relations | Written for this project in the style of *Hadashot Arkheologiyot*. They are not real reports | This repository | [`02_layout_pipeline/examples/`](../02_layout_pipeline/examples) |
| **Generated images** | 600 images, 300 layouts | Stable Diffusion 1.5 + ControlNet-seg, [`build_synthetic.py`](../04_cv_experiment/src/build_synthetic.py) | CreativeML OpenRAIL-M applies to the model | Layouts, labels and `manifest.csv` in [`04_cv_experiment/data/synthetic/`](../04_cv_experiment/data/synthetic); images in release [`data-v1`](https://github.com/ssen2803/arch-to-image/releases/tag/data-v1) |
| **Real drone tiles** | 56 tiles, 800×800, wall polygons | Uziel, Zohar and Jaffe 2026, [10.5281/zenodo.15673668](https://doi.org/10.5281/zenodo.15673668) | MIT | Downloaded; samples in [`samples/real_walls/`](samples/real_walls) |
| **Report and plan pairs** | Not collected yet | *Hadashot Arkheologiyot*, Israel Antiquities Authority | © IAA, written consent requested 14 Nov 2026 | Not in the repository |

## Generated images

- **Process.** Random layout (`synth_layouts.py`), checked by the validator of stage 2 → class mask → ADE20K colours → ControlNet, 30 UniPC steps, guidance 7.5 → 512×512 image. The mask is the label.
- **Controlled attributes.** Period (7), context (building or open area), element types (12), sizes in metres, doorways, light (4), diffusion seed (2).
- **Stored per image** in `manifest.csv`: image, label, layout id, seed, prompt, and scores of an external segmenter (`agree_miou`, `HOR_area`, `object_recall`, `HAR`).
- **Filtering.** Black images from the safety filter are retried with seed + 1000. Nothing else is filtered: the two available quality scores disagree with each other (see the EDA).
- **Splits, by layout number.** Ends in 0: validation (60 images). Ends in 5: generated test (60). The rest: training (480).
- **Known limitations.** Noisy labels; colours about 3.6 times more saturated than the real development tiles; walls about twice as thick in the frame; only two diffusion seeds, so 59 of 60 generated test images share a seed with training images. Details: [`notebooks/01_data_exploration.ipynb`](../notebooks/01_data_exploration.ipynb).

## Real drone tiles

```bash
# 552 MB, md5 8a60023104eb2658850b41f0afc5651e
curl -L -o StonesTracker-project.zip "https://zenodo.org/api/records/15673668/files/StonesTracker-project.zip/content"
# extract annotated_tiles/ to 04_cv_experiment/data/real/uziel_extracted/, then:
python 04_cv_experiment/src/prepare_real_walls.py
```

- **Labels.** Wall segments only. Four spellings of the label name in the source all mark walls and are merged. Black no-data areas are ignored.
- **Leakage.** Among the 54 tiles used, 30 pairs overlap (17 by a quarter, 13 at a corner). Two more tiles are second copies of the same photo, labelled again. Tiles are grouped by overlap and split by group: 14 development, 40 test, 2 duplicates removed. Manifest: [`real_walls_manifest.csv`](../04_cv_experiment/annotation/real_walls_manifest.csv).
- **Label consistency.** On the two photos labelled twice, the wall IoU between the two labels is 0.73 and 0.69. Two photos only, annotators unknown.
- **Fingerprint.** SHA-256 of every test image and mask: [`real_walls_test.sha256`](../04_cv_experiment/annotation/real_walls_test.sha256). Deviations from the planned protocol: [`test_set_protocol.md`](../04_cv_experiment/test_set_protocol.md), section 8.
- **Known limitations.** The source does not say which site a tile comes from, so a split by site is not possible. Only walls are labelled.
- **Credit.** Erel Uziel, Motti Zohar, Yitzchak Jaffe, *Semi-automatic detection of building stones and wall segments of archaeological ruins*, Journal of Archaeological Science 185, 2026, 106430.
