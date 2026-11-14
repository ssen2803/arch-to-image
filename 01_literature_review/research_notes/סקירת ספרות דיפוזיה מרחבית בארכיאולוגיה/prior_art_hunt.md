# Adversarial Prior-Art Hunt: text -> structured spatial layout -> layout-conditioned diffusion -> CV evaluation, on archaeological / heritage data

Notation used throughout. The project's four stages are:
- **S1**: heritage/archaeological *text* as input (excavation report, historical record).
- **S2**: an *LLM* turns the text into a *structured spatial layout* (boxes, plan, segmentation).
- **S3**: *layout-conditioned diffusion* (ControlNet, GLIGEN or similar) renders the image.
- **S4**: *computer-vision evaluation of layout faithfulness* (a detector or segmenter is run on the output and scored with IoU, AP or mIoU against the input layout).

Search date: 2026-11-14. Coverage window: mainly 2023 to November 2026.

## Which searches were run, in which databases, and what did each return? (search log)

### Takeaway
I ran about 130 distinct queries across seven channels: a general web engine, the arXiv API, OpenAlex, Semantic Scholar, the GitHub API, Europe PMC/Crossref, and a Chinese-language web query. No query returned a work that implements all four stages on archaeological or heritage data. Some returned heritage works covering 2 to 3 stages, and generic-domain works covering all 4. Several important channels were inaccessible or only partly accessible (Google Scholar, ProQuest, CNKI, paywalled full texts); see Gaps.

### Cited Findings

**A. General web search engine (WebSearch tool; "extended" mode unless noted).** Each line gives the query, then the relevant hits.
1. `archaeological reconstruction diffusion model layout ControlNet`: no direct match. It surfaced the archaeological line-drawing paper (FLUX.1 Kontext, image to drawing) and the Qing tea-plantation SD+ControlNet paper — [Generating archaeological line drawings](https://www.nature.com/articles/s40494-026-02526-3); [Qing tea plantations](https://www.nature.com/articles/s40494-025-02183-y)
2. `excavation report text to image generative AI archaeology`: no report-to-image system. Hits were DALL-E illustration studies, PyPotteryInk and "From ruins to reconstruction" — [Magnani & Clindaniel 2023](https://doi.org/10.1017/aap.2023.25); [From ruins to reconstruction](https://doi.org/10.20528/cjsmec.2024.02.004)
3. `heritage reconstruction LLM layout diffusion arXiv`: no heritage match. The closest LLM-then-diffusion hit was HouseLLM/HouseTune (floor plans) — [arXiv 2411.12279](https://arxiv.org/abs/2411.12279)
4. `ancient architecture generation ControlNet cultural heritage`: returned the multi-agent Chinese-architecture paper, the historical-facade paper, the Minnan-facade paper and the Qing tea paper — [Lu et al. 2025](https://doi.org/10.1038/s41598-025-18130-7); [Kuang et al. 2025](https://doi.org/10.1038/s40494-025-01826-4); [Xu et al. 2024](https://doi.org/10.3390/info15060344)
5. `"archaeological site" text-to-image GLIGEN OR "layout-to-image"`: 0 heritage hits, only generic GLIGEN material — [GLIGEN](https://arxiv.org/abs/2301.07093)
6. `LLM generated layout historical scene image generation bounding boxes`: generic LLM-to-layout-to-diffusion papers only (LMD, LLM Blueprint, Koch et al. 2025) — [LMD](https://arxiv.org/abs/2305.13655); [Koch et al.](https://arxiv.org/abs/2511.06888)
7. `site plan to image diffusion ruins reconstruction`: no plan-to-image heritage work. The Suoyang virtual-reconstruction paper came up — [Suoyang](https://doi.org/10.1038/s40494-025-02266-w)
8. `museum artifact text description layout-conditioned diffusion evaluation object detection`: no heritage match. It returned generic layout benchmarks — [Parolari et al. 2026](https://arxiv.org/abs/2604.25358)
9. `generative reconstruction archaeological site plan Stable Diffusion ControlNet segmentation`: surfaced the Huizhou knowledge-graph-to-ControlNet chapter and the JOCCH heritage-pattern paper — [Guo et al.](https://doi.org/10.1007/978-981-92-0615-5_35); [Liu et al. JOCCH](https://doi.org/10.1145/3789209)
10. `text-to-floorplan Tell2Design ChatHouseDiffusion HouseLLM evaluation IoU`: floor-plan works with IoU evaluation (modern housing) — [HouseMind comparison table](https://arxiv.org/html/2603.11640)
11. `LLM archaeological text structured spatial layout generation excavation`: no archaeological layout generation. A CIDOC-CRM text-to-triples workshop paper came up — [From Text to Triples](https://ceur-ws.org/Vol-4099/ER25_DC_hariri.pdf)
12. `Pompeii Roman house reconstruction diffusion floor plan conditioned image generation`: none. Only older shape-grammar Roman-house work and modern floor-plan diffusion — [Semi-automatic reconstruction of Roman houses](https://www.researchgate.net/publication/261042594_Semi-automatic_virtual_reconstruction_of_ancient_roman_houses)
13. `text-to-3D heritage site LLM procedural ancient city reconstruction`: none for heritage. Returned CityGenAgent (modern) and procedural historical cities (no LLM) — [CityGenAgent](https://arxiv.org/pdf/2602.05362); [4D historical cities](https://doi.org/10.3390/rs15133352)
14. `historical map generation diffusion model layout conditioned`: none for historical maps.
15. `e-Heritage workshop ICCV 2025 diffusion generation papers`: could not find the e-Heritage 2025 accepted-paper list.
16. `VISART ECCV 2024 workshop accepted papers diffusion` (standard): confirmed VISART VII. The only diffusion paper found was DIFF-NST (style transfer). Full list not retrieved — [ECCV 2024 workshops](https://eccv2024.ecva.net/Conferences/2024/Workshops)
17. `paleoart diffusion model reconstruction scientific accuracy evaluation` (standard): no paleoart diffusion study.
18. `crime scene reconstruction from witness text LLM layout diffusion` (standard): VeriScene — [arXiv 2609.08342](https://arxiv.org/abs/2609.08342)
19. `dataset excavation reports text paired with site plans machine learning archaeology`: no public text-plus-plan dataset found.
20. `"Suoyang" virtual reconstruction ...` (standard): found the Heritage (MDPI) 2026 pragmatic review and the Gonghoulong ISPRS 2025 paper — [Lombardi et al. 2026](https://doi.org/10.3390/heritage9070246); [Li et al. 2025](https://doi.org/10.5194/isprs-archives-XLVIII-M-9-2025-843-2025)
21. `generative AI archaeological reconstruction expert evaluation protocol archaeologists rating images`: expert-rating studies (Bamiyan, Harappan pigments, Iberian pottery, Roman coins) — [Bamiyan evaluation](https://zenodo.org/records/20435265)
22. `Hadashot Arkheologiyot machine learning dataset NLP reports` (standard): no ML/NLP dataset. Only a statistical settlement dataset — [Pandora dataset](https://pandoradata.earth/en/dataset/late-andtiquity-early-islamic-settlements-in-israel-and-palestine)
23. `historical cadastral map to street view image generation diffusion`: none for cadastral maps. CrossViewDiff (historical satellite and OSM to street view) came up — [arXiv 2408.14765](https://arxiv.org/abs/2408.14765)
24. `考古 遗址 复原 扩散模型 ControlNet 布局 大语言模型` (Chinese: "archaeology, site, reconstruction, diffusion model, ControlNet, layout, LLM"): only ControlNet tutorials (CSDN/Zhihu), no research hits.
25. `"Digital Applications in Archaeology and Cultural Heritage" diffusion ControlNet reconstruction 2025` (standard): led to the HARF (Brunel) work — [BURA record](https://bura.brunel.ac.uk/handle/2438/33750)
26. `"Heritage-Aligned Reconstruction Framework" Arzomand Palmyra ...` (standard): HARF publications — [Heritage 2026](https://doi.org/10.3390/heritage9080331)
27. `LLM reads archaeological report generates site plan then image generation reconstruction`: the Gonghoulong ISPRS paper was the closest — [Li et al. 2025](https://doi.org/10.5194/isprs-archives-XLVIII-M-9-2025-843-2025)
28. `textual sources historical reconstruction image generation segmentation mIoU evaluation heritage`: no paper reports mIoU or AP on generated heritage reconstructions.

**B. arXiv API (`export.arxiv.org/api/query`, field `abs:` = title + abstract, sorted newest first).** 58 queries in 3 batches. Totals are the arXiv `totalResults`. Note that `archaeolog*` terms also match astrophysics "Galactic archaeology".
- Heritage core: `abs:archaeological AND abs:diffusion` (21; relevant hits: PyPotteryInk, ReassembleNet, Diff-Oracle, Knowledge-Aware Artifact Synthesis 2312.08056, porcelain SD augmentation); `abs:archaeology AND abs:"text-to-image"` (3; 2312.08056, Skull-to-Face, provenance RAG); `abs:archaeological AND abs:"large language model"` (11; no generation-with-layout work); `abs:excavation AND abs:generative` (172; all about machinery or physics, none archaeological); `abs:"cultural heritage" AND abs:ControlNet` (2; Ulos motifs 2609.17987, Land Art forecasting); `abs:heritage AND abs:layout AND abs:diffusion` (2; Ulos, CalliMaster 2603.12482); `abs:heritage AND abs:"text-to-image"` (13; no layout pipeline); `abs:historical AND abs:ControlNet` (1; arcade facades 2311.11590); `abs:ancient AND abs:ControlNet` (2; DiffuMural); `abs:ancient AND abs:layout AND abs:diffusion` (1; UniCalli); `abs:ruins AND abs:diffusion` (47; all actuarial "ruin probability"); `abs:"archaeological site" AND abs:reconstruction AND abs:generative` (2; irrelevant); `abs:museum AND abs:layout AND abs:diffusion` (0); `abs:"site plan" AND abs:diffusion` (1; GeoTexBuild); `abs:"floor plan" AND abs:ControlNet` (0); `abs:paleontology AND abs:diffusion` (2; Jurassic World Remake); `abs:"historical scene" AND abs:generation` (5; video world models, irrelevant); `abs:heritage AND abs:"scene graph" AND abs:generation` (0).
- Variants: `abs:"traditional architecture" AND abs:diffusion` (3; irrelevant); `abs:"historic" AND abs:"Stable Diffusion"` (20; RenAIssance 2611.10599, arcade facades); `abs:heritage AND abs:"Stable Diffusion" AND abs:reconstruction` (1; temple arts 2503.10031); `abs:"virtual reconstruction" AND abs:generative` (6; none relevant); `abs:Pompeii` (5; none generative); `abs:Roman AND abs:architecture AND abs:diffusion` (0); `abs:temple AND abs:diffusion AND abs:generation` (5; MVS/3D, not heritage); `abs:museum AND abs:"text-to-image"` (7; ArtiFact dataset 2606.09648, MuseumMaker); `abs:heritage AND abs:"large language model" AND abs:"image generation"` (2; Ulos, Indian poetry); `abs:"cultural heritage" AND abs:layout` (9; OCR/document layout, plus CalliMaster); `abs:"floor plan" AND abs:diffusion AND abs:render` (2; ReSpace, room assembly); `abs:floorplan AND abs:photorealistic` (2; PanoWorld 2605.17916); `abs:architectural AND abs:ControlNet` (75; noise, because "architectural" matches network-architecture abstracts); `abs:"layout-to-image" AND abs:medical` (2; retinal vessel 2503.01190); `abs:"scientific illustration" AND abs:diffusion` (2; SridBench, SciDraw-6K); `abs:"YOLO score" AND abs:layout` (0); `abs:"layout fidelity" AND abs:diffusion` (5; Koch et al. 2511.06888, UniCalli); `abs:"text-to-layout" AND abs:diffusion` (10; CalliMaster, Lay-Your-Scene, Text2Street, 7Bench, C-Bench); `abs:"historical" AND abs:"layout" AND abs:"large language model" AND abs:image` (8; StageDesigner 2503.02595, plus document-OCR papers); `abs:"ancient" AND abs:"text-to-image"` (4; Poetry2Image 2407.06196, 2312.08056); `abs:"lost" AND abs:heritage AND abs:generative` (3; irrelevant); `abs:"historical photographs" AND abs:diffusion` (0).
- Batch 3: `abs:"radiology report" AND abs:layout AND abs:diffusion` (0); `abs:"chest X-ray" AND abs:"bounding box" AND abs:"language model" AND abs:generation` (8; all report generation, not image generation); `abs:"floor plan" AND abs:interior AND abs:ControlNet` (0); `abs:floorplan AND abs:"text-to-image" AND abs:render` (0); `abs:heritage AND abs:agent AND abs:3D AND abs:generation` (1; VaseMuseum, a QA agent with no generation); `abs:ancient AND abs:"language model" AND abs:3D AND abs:reconstruction` (0); `abs:"historical" AND abs:"layout-to-image"` (0); `abs:"cultural heritage" AND abs:"large language model" AND abs:"diffusion"` (1; Ulos); `abs:archaeological AND abs:layout` (6; PyPottery suite, SARCH archive search, none generative-layout); `abs:"historical architecture" AND abs:generation` (1; practice essay); `abs:"heritage" AND abs:"floor plan"` (0); `abs:"ancient" AND abs:"floor plan"` (1; Versailles-FP wall-detection dataset 2103.08064); `abs:"text-to-scene" AND abs:historical` (0); `abs:"paleoart" OR abs:"palaeoart"` (0); `abs:"forensic" AND abs:"text-to-image"` (21; all deepfake forensics); `abs:"layout-guided" AND abs:"detector" AND abs:evaluation` (0).

**C. OpenAlex API (`api.openalex.org/works?search=`, filtered to 2022 onward).** It indexes journals the arXiv search misses: npj Heritage Science, MDPI Heritage, JOCCH, DAACH, Virtual Archaeology Review, ISPRS Archives and LNCS.
- `archaeological reconstruction ControlNet` (71): spatial-compliance inpainting framework, Roman frescoes (expert-guided diffusion), bronzeware spatial-compliance indices, Buddhist sculpture 3D. No text-to-layout work.
- `archaeological site reconstruction diffusion model image` (3,897): PyPotteryInk, TEXMET, ReassembleNet. None with layout.
- `excavation report large language model image generation` (5,996): only the LLM-in-archaeology review (Electronics 2025). No generation work.
- `heritage layout diffusion large language model` (2,685): textile/ceramic/kite generation. No spatial layout.
- `archaeology text-to-image generative evaluation` (4,658): AAP papers, "From ruins to reconstruction", "Architecture inside the mirage".
- `virtual archaeology generative AI site plan` (1,697): Ostia-Portus generative (procedural) tools, Cape Verde AI virtual reconstruction.
- `ancient building plan conditioned image generation` (29,549): noise (point clouds, MSD floor-plan dataset).
- `cultural heritage GLIGEN layout` (9): Thangka and mural generation. No spatial-plan work.
- `heritage ControlNet segmentation evaluation` (133): Qing tea plantations, Huizhou KAR chapter, the heritage-ornament evidence framework, Zhuang brocade.
- `Roman generative AI reconstruction image` (4,852): Roman coins GAN, the Saint Engracia crypt (VAR 2026), "GenAI images of Roman Britain".
- `archaeological illustration generative AI` (1,366): AAP 2023, Ostia-Portus, Internet Archaeology 2026 essay.
- `historical scene generation layout large language model` (11,644): noise.
- `Heritage-Aligned Reconstruction Framework Bamiyan generative` (12) and `evidence-constrained generative AI heritage reconstruction Bamiyan` (2): the HARF family (DAACH 2026, Heritage 2026, Zenodo/SSRN 2026, CJSMEC 2024).

**D. Semantic Scholar Graph API (`/paper/search`).** Of 10 queries, 6 failed with HTTP 429 (rate limit) even after retries. The 4 that returned were mostly noise. `archaeological reconstruction ControlNet` (12,614) was led by the spatial-compliance inpainting paper; `lost architecture reconstruction stable diffusion historical documents` (117) returned Minnan facades, Qing tea plantations and Buddhist sculpture. A batch DOI lookup also failed with 429.

**E. GitHub repository search API.** 10 queries. `archaeology controlnet` (0), `archaeology stable diffusion` (0), `archaeology layout generation` (0), `ruins reconstruction diffusion` (0) and `archaeology LLM image generation` (0) returned nothing. `heritage controlnet` (2: Heritage-VLM restoration, historical photo restoration). `heritage text-to-image` (7: e.g., `ahrbadr/Heritage-3D-Gen` text-to-image then 3D; `i-SAVE/krasny-treugolnik-mcda` T2I benchmark for industrial heritage; `Wasan-M-Alomar/mathlooth-saudi-heritage-3d`). `archaeological diffusion` (2: SherdFusion sherd reassembly). None implements text-to-layout-to-layout-conditioned diffusion — [GitHub search API](https://api.github.com/search/repositories?q=heritage+text-to-image)

**F. Metadata lookups.** Used Europe PMC (Lu et al. 2025, PMC12494968), Crossref (Huizhou chapter, JOCCH paper) and OpenAlex DOI lookups (17 DOIs) to verify authors, years and abstracts. Nature.com, link.springer.com, mdpi.com, dl.acm.org and Eurographics diglib returned login redirects or 403, so I could not read their full texts.

### Inferences
- The search space is well covered for English preprints and indexed journals from 2023 to 2026. The absence of a four-stage heritage hit across arXiv, OpenAlex, the web engine and GitHub is meaningful evidence, not just a failure to search.
- Several "hits" are lexical false positives (Galactic archaeology, ruin probability, excavator robotics, network "architecture"). Raw result counts therefore overstate how much heritage generative work exists.

### Gaps
- **Google Scholar, ProQuest dissertations, CNKI/Wanfang (Chinese) and J-STAGE were not queried directly**, because the tools had no access. The only Chinese query went to the general web engine and returned tutorials only. Much heritage-plus-ControlNet work comes from Chinese groups, so this is the largest residual blind spot.
- The arXiv API searches title and abstract, not full text. A paper that mentions ControlNet or an LLM only in its methods section would be missed.
- Workshop proceedings were not enumerated: ICCV 2025 e-Heritage, ECCV 2024 VISART VII, AI4CH, Eurographics GCH, CAA 2024–2026 and CIPA 2025 beyond the one ISPRS paper found.
- The Semantic Scholar channel was mostly lost to rate limiting.

## Does any work cover 3 or 4 of the 4 stages on heritage, historical or archaeological data?

### Takeaway
No work found implements all four stages on heritage data. At most 2 to 3 stages are covered, often only conceptually. Four works come closest, each in a different way:
- **Gonghoulong ISPRS 2025**: an LLM with retrieval reads the site plan and literature, infers the structure, then a generative model renders. It is described at framework level, has no CV evaluation, and the model is not even named.
- **Huizhou KAR chapter, 2026**: a knowledge graph, not an LLM, drives ControlNet conditioning. The outputs are facades, not site layouts.
- **HARF, 2024–2026**: archaeological evidence is turned into structured prompts, and outputs get a geometric pre-screen plus expert panels. There is no layout-conditioned diffusion.
- **CalliMaster, 2026**: a text -> layout boxes -> image pipeline with OCR-based evaluation. Its domain is cultural-heritage calligraphy, not spatial scenes, and an LLM does not produce the layout.

### Cited Findings

**Stage-coverage matrix. Y = covered, P = partial or indirect, N = absent.**

| Work | S1 heritage text | S2 LLM spatial layout | S3 layout-cond. diffusion | S4 CV layout eval | Source |
|---|---|---|---|---|---|
| Li, Pan, Han, Gu, Ding, Tang, Liu (2025), "Intelligent Deduction and Historical Reconstruction an AIGC-Driven Exploration of the Main Hall Ruins at Gonghoulong Temple, Ningde, China", ISPRS Archives XLVIII-M-9-2025, 843–849 | Y | P | P | N | [DOI](https://doi.org/10.5194/isprs-archives-XLVIII-M-9-2025-843-2025) |
| Guo, Xiong, Zhang, Kong (2026), "Knowledge-Augmented Stable Diffusion Model for Semantic Representation of Architectural Heritage: A Case Study of Huizhou Traditional Dwellings", in *Transindividual Intelligence* (CDRF 2025), pp. 410–419 | P | P | Y | P | [DOI](https://doi.org/10.1007/978-981-92-0615-5_35) |
| Arzomand, Kalganova, Rustell (2026), "HARF: A human–AI collaborative framework for cultural heritage reconstruction with expert-guided multi-platform generative AI and systematic prompt engineering", DAACH | Y | P | N | P | [DOI](https://doi.org/10.1016/j.daach.2026.e00554) |
| Xu, Hong, Chen, Chao, Chen (2026), "CalliMaster: Mastering Page-level Chinese Calligraphy via Layout-guided Spatial Planning", arXiv 2603.12482 | P | P | Y | P | [arXiv](https://arxiv.org/abs/2603.12482) |
| Lu, Yuan, Wang, Wang, Wu, Wu, Xing, Xie, Yu (2025), "Multi-agent collaborative pathways for Chinese traditional architectural image generation", Sci. Rep. 15 | P | N | P | N | [DOI](https://doi.org/10.1038/s41598-025-18130-7) |
| Artal-Sanz, Gutiérrez Pérez, Chías Navarro (2026), "Application of generative AI to realistic image rendering in virtual reconstruction: the crypt of the Royal Monastery of Saint Engracia in Zaragoza", Virtual Archaeology Review | P | P (manual) | P | N | [DOI](https://doi.org/10.4995/var.2026.25819) |
| Wu, Chen, Su (2023), "Knowledge-Aware Artifact Image Synthesis with LLM-Enhanced Prompting and Multi-Source Supervision", arXiv 2312.08056 | Y | N | N | P | [arXiv](https://arxiv.org/abs/2312.08056) |
| Chen, Dai, Zhang, Zeng (2026), "Multimodal intelligent reconstruction-driven digital regeneration of cultural heritage in Qing Dynasty tea plantation landscapes", npj Heritage Science | N | P | Y | N | [DOI](https://doi.org/10.1038/s40494-025-02183-y) |
| Zhang, Luo, Xu (2026), "An evidence-informed fine-tuning framework for traceable generative redesign of architectural heritage ornaments", npj Heritage Science | P | P | P | P | [DOI](https://doi.org/10.1038/s40494-026-03053-x) |

**Details for each row:**
- **Gonghoulong (ISPRS 2025).** This is the most dangerous paper for the novelty claim at the concept level.
  - The paper describes an LLM "fine-tuned on Song architecture and archaeology" combined with vector retrieval over a knowledge base. When inferring the column grid or roof type, the system retrieves analogous Song-dynasty temple plans and construction rules — [Li et al. 2025, PDF §4.1](https://isprs-archives.copernicus.org/articles/XLVIII-M-9-2025/843/2025/isprs-archives-XLVIII-M-9-2025-843-2025.pdf)
  - The paper states that the AI system first "reads" the partial site plan and point-cloud geometry, infers the complete column grid and roof form, and then generates both a textual restoration report and 3D/visual renderings — [same PDF](https://isprs-archives.copernicus.org/articles/XLVIII-M-9-2025/843/2025/isprs-archives-XLVIII-M-9-2025-843-2025.pdf)
  - The generator is described only as "a deep generative model (e.g. a diffusion model or GAN)". It is trained on imagery of Song-dynasty halls, archaeological illustrations and existing reconstructions, and given the inferred 3D framework it produces plan views, elevations and renderings. No model, no ControlNet and no quantitative image metric is named — [same PDF](https://isprs-archives.copernicus.org/articles/XLVIII-M-9-2025/843/2025/isprs-archives-XLVIII-M-9-2025-843-2025.pdf)
  - Validation is a "three-dimensional" comparison of the digital (BIM) model with reference temple models and historical records, with discrepancies reviewed by experts. This is not a CV evaluation of the generated images — [same PDF](https://isprs-archives.copernicus.org/articles/XLVIII-M-9-2025/843/2025/isprs-archives-XLVIII-M-9-2025-843-2025.pdf)
- **Huizhou KAR chapter.**
  - The authors encode the hierarchical layout, materials and ornaments of Huizhou dwellings into a shape grammar and build a Neo4j property-graph knowledge base that supports "reasoning, retrieval, and prompt generation". An MMDiT incorporates "KAR-guided ControlNet instances" — [Guo et al. 2026 abstract via OpenAlex](https://doi.org/10.1007/978-981-92-0615-5_35)
  - Evaluated on facades of Xidi Ancient Village, the model is reported to beat baselines in "style consistency, efficiency, and structural fidelity". The abstract does not name the metric — [same](https://doi.org/10.1007/978-981-92-0615-5_35)
  - A search-engine summary said the graph reasoning "dynamically generate[s] ControlNet constraint masks". I could not verify this against the full text, which is paywalled — [search hit, Springer page](https://link.springer.com/chapter/10.1007/978-981-92-0615-5_35)
- **HARF family (Brunel).**
  - HARF has three components: a blueprint encoding verified archaeological and iconographic information into prompts; a quantitative prompt-completeness index; and a "geometric pre-evaluation stage that screens outputs against site-specific tolerances". It was validated with 33 international specialists on the Western Buddha of Bamiyan — [Arzomand et al. 2026, DAACH](https://doi.org/10.1016/j.daach.2026.e00554)
  - The Palmyra extension refilled the blueprint with evidence on the temple's dimensions, Corinthian order and "architectural layout". 32 panel members rated outputs, and 56.2% selected "none of the iterations are historically accurate" for relief iconography — [Arzomand & Kalganova 2026, Heritage 9(8):331](https://doi.org/10.3390/heritage9080331)
  - The Bamiyan evaluation preprint covers 245 candidates from "two generative platforms", screened by "structural similarity and feature-matching metrics" and then a structured expert rubric — [Zenodo 10.5281/zenodo.20435265](https://zenodo.org/records/20435265)
  - The precursor study used text prompts built from historical records with Midjourney, DALL-E and Stable Diffusion, and reported SSIM, MSE, PSNR and MAE — [Arzomand, Rustell, Kalganova 2024](https://doi.org/10.20528/cjsmec.2024.02.004)
  - No ControlNet or layout-conditioning step appears in any HARF abstract I read.
- **CalliMaster.**
  - The paper describes an explicit "(Text → Layout → Image)" pipeline. A spatial-planning stage predicts character bounding boxes, and the layout then serves as a geometric prompt for flow-matching synthesis. The paper frames itself as "a comprehensive tool for digital cultural heritage" — [arXiv 2603.12482](https://arxiv.org/abs/2603.12482)
  - The layout is predicted by the same MM-DiT, not by an LLM. The input text is the character content plus style metadata, not a descriptive report — [arXiv HTML](https://arxiv.org/html/2603.12482v1)
  - Evaluation uses CLIP-Score, GPT-Score, an OCR-based CharsAcc (PP-OCR F1) and a user study including "Layout Similarity". No box-IoU metric is reported — [same](https://arxiv.org/html/2603.12482v1)
- **Multi-agent Chinese architecture (Sci. Rep. 2025).**
  - User text goes through agents, including DeepSeek-V3 for prompt optimisation, to SD with LoRA plus a Canny ControlNet. The knowledge base (built from Yingzao Fashi and other sources) encodes spatial relations as symbolic predicates, not layouts — [PMC12494968](https://pmc.ncbi.nlm.nih.gov/articles/PMC12494968/)
  - Evaluation is CLIP Score (0.89) plus a 60-person Likert study (20 experts). There is no quantitative layout metric — [same](https://pmc.ncbi.nlm.nih.gov/articles/PMC12494968/)
- **Saint Engracia crypt (VAR 2026).**
  - Historical documentation and architectural analogues led to a *manual* floor-plan hypothesis in AutoCAD, then a Rhino 3D model, then an untextured D5 render. That render was then processed with Gemini's Nano Banana Pro to generate textures and wall paintings.
  - The authors found that GenAI "introduce[s] arbitrary artefacts" and requires human supervision — [Artal-Sanz et al. 2026](https://doi.org/10.4995/var.2026.25819)
  - This is the closest heritage example of "plan -> structure-conditioned generation", but S2 is manual and S4 is qualitative.
- **Knowledge-aware artifact synthesis.** The method builds prompts with "explicit archaeological knowledge elicited from LLMs" and adds edge and perceptual constraints during training. It reports gains on automatic metrics and human evaluation. It generates single artifacts with no spatial layout — [arXiv 2312.08056](https://arxiv.org/abs/2312.08056)
- **Qing tea plantations.** Semantic segmentation and HSV analysis of 104 Qing export paintings drive "Stable Diffusion + ControlNet". The input is images, not text — [Chen et al. 2026](https://doi.org/10.1038/s40494-025-02183-y)
- **Heritage ornaments.** An "evidence-to-conditioning translation layer" converts source-audited evidence into ontology-linked semantic units, structured prompt fields, exclusion fields and relation-check configurations. Across 700 outputs the paper reports an expert mean of 4.66, SSIM 0.982 and LPIPS 0.041. The subject is ornaments, not spatial layout — [Zhang et al. 2026](https://doi.org/10.1038/s40494-026-03053-x)
- **Other heritage diffusion works with at most 1–2 stages:**
  - Historical arcade facades (LoRA plus ControlNet, text prompts) — [Kuang et al. arXiv 2311.11590](https://arxiv.org/abs/2311.11590); [npj Heritage Science 2025](https://doi.org/10.1038/s40494-025-01826-4)
  - Minnan facades: Grasshopper line drawings feed ControlNet lineart plus LoRA, so the layout is procedural, not from an LLM — [Xu, Zhang, Li 2024](https://doi.org/10.3390/info15060344)
  - Heritage patterns: ControlNet plus LoRA with AHP expert assessment — [Liu, Li, Li, Wang 2026, JOCCH](https://doi.org/10.1145/3789209)
  - Ulos motifs: MLLM plus SDXL LoRA plus a semantic-map ControlNet, evaluated with FID, CLIP, SSIM and weaver ratings — [Simanjuntak et al. 2026](https://arxiv.org/abs/2609.17987)
  - Ostia-Portus: procedural Blender geometry-nodes "study models" from the excavation stage, with no diffusion — [Fabro & Saleri 2026](https://doi.org/10.6093/2284-4732/13379)
  - Suoyang: drone, laser scanning, VR/AR and AI virtual reconstruction, with no layout-conditioned diffusion in the abstract — [Liu et al. 2026](https://doi.org/10.1038/s40494-025-02266-w)
- **Field-level context.** A systematic review found that, of over 8,700 papers on 3D heritage reconstruction from 2015 to 2024, "only 13 directly addressed generative AI-driven reconstruction processes". It also flagged "evaluation metrics focused on visual plausibility rather than scientific transparency" — [Lombardi, Masini, Abate 2026, Heritage 9(7):246](https://doi.org/10.3390/heritage9070246)

### Inferences
- The Gonghoulong paper is the work a reviewer would most likely cite against an "archaeological text -> plan/structure -> generative rendering" novelty claim. It describes that idea in prose. But it gives no implemented, named layout-conditioned diffusion model, no released code, and no CV-based faithfulness evaluation. The project's defensible delta is an implemented, reproducible S2→S3 handoff plus a quantitative S4.
- HARF is the work a reviewer would cite against the "rigorous evaluation of heritage reconstructions" part. It already has evidence-structured prompting, geometric tolerance screening and expert panels of over 30 people. The project should position its CV metrics (detector or segmenter IoU against the input plan) as complementary to HARF's expert protocol, not as the first evaluation of heritage generative reconstruction.
- CalliMaster shows that "text -> layout -> layout-conditioned diffusion -> CV check" has already appeared under a cultural-heritage label, albeit for calligraphy pages. The novelty should be stated as *spatial-scene / site-plan* layouts derived from *excavation-report prose* by an *LLM*, not "first layout-guided generation for heritage".

### Gaps
- Full texts of the Huizhou chapter, HARF DAACH, the Brunel PhD thesis ([BURA 2438/33949](https://bura.brunel.ac.uk/handle/2438/33949)) and the Qing tea paper could not be read. Any of them could contain an LLM-to-mask step or a segmentation-based metric that is not visible in the abstracts.
- The Gonghoulong paper's actual generative model and its outputs were not identified. The PDF text says only "e.g. a diffusion model or GAN".

## What are the closest analogues in adjacent domains that a reviewer would cite?

### Takeaway
In the general (non-heritage) domain, the full four-stage pipeline already exists and is well cited:
- **LayoutGPT**: LLM layout -> GLIGEN -> GLIP detector accuracy.
- **VPGen + VPEval**: an LM generates the layout, followed by image generation, then evaluation with expert visual modules.
- **LMD and LLM Blueprint**: LLM boxes followed by layout-grounded SD.

Domain-specific analogues include floor plans (Tell2Design, ChatHouseDiffusion, HouseTune: text -> layout with IoU evaluation, but the plan is the final output), stage scenography (StageDesigner: scripts -> GPT-4o layout -> layout-controlled diffusion, including historical styles) and street scenes (Text2Street). None of these is heritage. So the *method* is not novel. Only the *domain instantiation and evaluation protocol* can be.

### Cited Findings
- **LayoutGPT (Feng, Zhu, Fu, Jampani, Akula, He, Basu, Wang, Wang; NeurIPS 2023).** LLMs generate layouts from text, which are combined with a downstream image generator — [arXiv 2305.15393](https://arxiv.org/abs/2305.15393). GLIGEN is the main layout-to-image model; GLIP (fine-tuned on COCO) detects objects in the generated images to compute "Acc. (GLIP)" for counting and spatial relations on the NSR-1K benchmark — [arXiv HTML](https://arxiv.org/html/2305.15393v2). **Covers S2+S3+S4 in full (generic COCO domain).**
- **VPGen / VPEval (Cho, Zala, Bansal; 2023).** An LM handles object/count generation and layout generation, followed by image generation. VPEval invokes "a set of visual modules that are experts in different skills" to evaluate the output — [arXiv 2305.15328](https://arxiv.org/abs/2305.15328). **All four stage types, generic domain.**
- **LLM-grounded Diffusion (Lian, Li, Yala, Darrell; 2023).** An LLM generates captioned bounding boxes, then a controller guides off-the-shelf diffusion for layout-grounded generation. It reports "doubling the generation accuracy across four tasks" — [arXiv 2305.13655](https://arxiv.org/abs/2305.13655)
- **LLM Blueprint (ICLR 2024).** Scene blueprints containing object boxes, per-box descriptions and a background prompt for long, complex prompts — [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/adf7fa39d65e2983d724ff7da57f00ac-Paper-Conference.pdf)
- **Koch, Krumme, Gadzicki (2025), "A Two-Stage System for Layout-Controlled Image Generation using Large Language Models and Diffusion Models".**
  - An LLM produces a structured layout, and ControlNet is compared head-to-head with GLIGEN after fine-tuning on table-setting data. ControlNet "suffers from object hallucination" while GLIGEN "provides superior layout fidelity" — [arXiv 2511.06888](https://arxiv.org/abs/2511.06888)
  - Image-level layout fidelity was judged only by visual inspection, with no detector, IoU or mAP — [arXiv HTML](https://arxiv.org/html/2511.06888v2)
  - This is the closest *design* match to a project comparing ControlNet and GLIGEN. It lacks S4.
- **Lay-Your-Scene / LayouSyn (Srivastava, Zhang, Wen, Wen, Tu; ICCV 2025).** Lightweight open-source LMs extract scene elements, and an aspect-aware diffusion transformer generates the layout — [arXiv 2505.04718](https://arxiv.org/abs/2505.04718)
- **Evaluation benchmarks for layout-guided diffusion (S4 machinery).**
  - LayoutBench tests number, position, size and shape skills, with out-of-distribution layouts — [Cho et al. 2023, arXiv 2304.06671](https://arxiv.org/abs/2304.06671)
  - C-Bench/O-Bench combine semantic and spatial accuracy into one score over 319,086 generated images from six layout-guided models — [Parolari, Faccioli, Ballan 2026, arXiv 2604.25358](https://arxiv.org/abs/2604.25358)
  - GLIGEN itself reports layout-to-image zero-shot results on COCO and LVIS — [arXiv 2301.07093](https://arxiv.org/abs/2301.07093)
- **Text-to-floor-plan works.** These cover S1+S2 and an IoU-type S4, but stop at the plan, with no rendering stage.
  - Tell2Design: a dataset of more than 80k floor plans paired with natural-language instructions, plus a Seq2Seq baseline and human evaluation — [Leng et al. 2023, arXiv 2311.15941](https://arxiv.org/abs/2311.15941)
  - ChatHouseDiffusion: an LLM interprets natural language, a graphormer encodes topology, and diffusion generates and edits floor plans, with higher IoU than prior models — [Qin et al. 2024, arXiv 2410.11908](https://arxiv.org/abs/2410.11908)
  - HouseTune / HouseLLM: an LLM with chain-of-thought produces "Layout-Init", which a conditional diffusion model refines — [Zong et al. 2024, arXiv 2411.12279](https://arxiv.org/abs/2411.12279)
  - HouseDiffusion: graph-conditioned vector floor-plan diffusion on RPLAN — [Shabani, Hosseini, Furukawa 2022, arXiv 2211.13287](https://arxiv.org/abs/2211.13287)
  - House-GAN++: graph-constrained GAN layout refinement — [Nauata et al. 2021, arXiv 2103.02574](https://arxiv.org/abs/2103.02574)
  - Reported micro-IoU: HouseMind-G 0.709 vs. ChatHouseDiffusion 0.589 vs. Tell2Design 0.390 (as reproduced). Tell2Design's own paper reports 54.34% micro-IoU, so cross-paper comparisons are inconsistent — [HouseMind, arXiv 2603.11640](https://arxiv.org/html/2603.11640); [HouseTune AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38417/42379)
- **StageDesigner (Gan, Li, Chen, Ji, Guo, Hu, Ye, Hu; 2025).**
  - Described by its authors as "the first comprehensive framework for artistic stage generation using large language models combined with layout-controlled diffusion models". Its StagePro-V1 dataset has 276 stage scenes "spanning different historical styles" annotated with scripts, images and 3D layouts — [arXiv 2503.02595](https://arxiv.org/abs/2503.02595)
  - GPT-4o extracts spatial cues and anchor-entity coordinates, and ReCo (layout-controlled diffusion) renders the background. Evaluation uses geometric layout metrics (OOB, OIS, IWG), CLIP-sim and user studies with 14 experts. No CV model checks layout adherence — [arXiv HTML](https://arxiv.org/html/2503.02595v1)
  - This is the strongest "text from a humanities source -> LLM layout -> layout-conditioned diffusion" analogue. It lacks a CV-based S4.
- **Architecture: plan or sketch to render.**
  - Sketch-based facade renovation: a fine-tuned VLM predicts bounding boxes for modifications, SD generates element sketches, and ControlNet refines the result to a photorealistic image. This is VLM-box-to-ControlNet in industrial adaptive reuse — [Booranamaitree et al. 2026, arXiv 2601.08531](https://arxiv.org/abs/2601.08531)
  - GeoTexBuild: footprints from site planning or map designs go through a customized ControlNet height map, then 3D buildings — [Wang, Yang, Wang 2025, arXiv 2504.08419](https://arxiv.org/abs/2504.08419)
  - PanoWorld: a floorplan-derived 3D shell leads to consistent whole-house panoramas — [Jia et al. 2026, arXiv 2605.17916](https://arxiv.org/abs/2605.17916)
- **Street and remote-sensing scenes.**
  - Text2Street: text goes to a road-topology map and a bounding-box layout (via box diffusion), then to a multi-control street-view image — [Su et al. 2024, arXiv 2402.04504](https://arxiv.org/abs/2402.04504)
  - GeoDiffusion: boxes and camera views are encoded as text for layout-to-image detection-data generation, with gains in detector performance — [Chen et al. 2023, arXiv 2306.04607](https://arxiv.org/abs/2306.04607)
  - CrossViewDiff: satellite and OSM inputs give street-view synthesis, applied to historical satellite imagery, with GPT-based scoring — [Li et al. 2024, arXiv 2408.14765](https://arxiv.org/abs/2408.14765)
- **Medical.** XReal adds anatomy and pathology spatial controllers to a pre-trained T2I diffusion model and is evaluated with quantitative metrics and radiologists' ratings. It has no LLM layout stage — [Hashmi et al. 2024, arXiv 2403.09240](https://arxiv.org/abs/2403.09240). The arXiv query `"radiology report" AND layout AND diffusion` returned 0 papers.
- **Forensic reconstruction.** VeriScene fuses forensic photos and witness statements into a cited narrative, then a world model renders a re-enactment video. It reports evidence coverage of 0.9014 and factual consistency of 0.7217. It has no 2D layout and no detector-based evaluation — [Fu, Zheng, Yeong, Lam 2026, arXiv 2609.08342](https://arxiv.org/abs/2609.08342)
- **Heritage text-to-image with LLM correction.** Poetry2Image uses an LLM-driven feedback loop to correct images generated from classical Chinese poetry. It reports 70.63% element completeness, but has no layout stage — [Jiang et al. 2024, arXiv 2407.06196](https://arxiv.org/abs/2407.06196)
- **Paleontology / scientific illustration.** "Jurassic World Remake" does zero-shot image-to-image translation from fossils to living animals — [arXiv 2308.07316](https://arxiv.org/abs/2308.07316). SridBench benchmarks scientific illustration drawing — [arXiv 2505.22126](https://arxiv.org/abs/2505.22126). Neither has a text -> layout stage.

### Inferences
- A reviewer will almost certainly cite LayoutGPT, LMD and VPGen/VPEval to show that "LLM -> layout -> GLIGEN -> detector evaluation" is an established recipe. The project must not claim *methodological* novelty for the pipeline itself.
- The defensible contributions are:
  1. The source text is long, technical excavation-report prose, not short captions.
  2. The layout vocabulary is archaeological (walls, loci, installations, plan conventions), not COCO classes. Off-the-shelf detectors are likely to fail on it, so S4 itself becomes a research problem.
  3. The application and evaluation are run on real archaeological data.
- StageDesigner and Koch et al. are the best templates to cite as "closest method, different domain". Koch et al. in particular already compares ControlNet with GLIGEN for layout fidelity.

### Gaps
- I did not verify LMD's evaluation detector or VPGen's exact image generator (whether GLIGEN) from the full texts, only from the abstracts.
- I did not run dedicated queries for "FloorplanDiffusion" or for paleoart-specific diffusion apps outside arXiv.
- I did not run full-text searches of medical layout-to-image work beyond arXiv abstracts.

## Has anyone evaluated generated heritage reconstructions with detectors/segmenters (mIoU/AP) against an input plan, or with formal expert-archaeologist protocols?

### Takeaway
I found **no** heritage work that runs a detector or segmenter on generated reconstructions and scores mIoU or AP against an input plan or layout. The closest CV-style evaluations are:
- HARF's geometric pre-screen against site tolerances, plus structural-similarity and feature-matching against archival imagery.
- Pixel and SSIM metrics against photos.
- "Spatial-compliance" mask-overflow metrics for ControlNet inpainting of archaeological images.

Formal expert protocols, by contrast, are common: HARF (33 and 32 experts), plus panels of 6–7 archaeologists in several artifact studies, and two architectural historians in an architecture study.

### Cited Findings
- **HARF.** A "geometric pre-evaluation stage that screens outputs against site-specific tolerances" with 33 specialists — [DAACH 2026](https://doi.org/10.1016/j.daach.2026.e00554). Structural-similarity and feature-matching metrics against archival imagery, then a structured rubric — [Zenodo 2026](https://zenodo.org/records/20435265). Palmyra was evaluated by 32 panelists using a 14-item questionnaire — [Heritage 2026](https://doi.org/10.3390/heritage9080331)
- **"From ruins to reconstruction".** SSIM, MSE, PSNR and MAE against reference images, plus iterative expert review — [CJSMEC 2024](https://doi.org/10.20528/cjsmec.2024.02.004); [ResearchGate summary](https://www.researchgate.net/publication/381416813_From_ruins_to_reconstruction_Harnessing_text-to-image_AI_for_restoring_historical_architectures)
- **Spatial-compliance framework.** Compares LaMa, SD Inpainting and ControlNet Inpainting on 35 photographs of Western Zhou vessels. It adds "response-based spatial compliance indicators" for damage coverage and out-of-mask overflow, where ControlNet's miscoverage ratio was 0.0450 vs. 0.0797 for SD — [Sui et al. 2026, Discover AI](https://doi.org/10.1007/s44163-026-01767-2). A related bronzeware paper proposes "spatial-compliance indices" — [npj Heritage Science 2026](https://doi.org/10.1038/s40494-026-02539-y). This is CV-based spatial-faithfulness evaluation in archaeology, but for inpainting, not layout-to-image.
- **CalliMaster.** An OCR-based (PP-OCR) character-accuracy metric on generated heritage calligraphy, plus a "Layout Similarity" user rating, with no box IoU — [arXiv HTML](https://arxiv.org/html/2603.12482v1)
- **Iberian pottery GAN.** Six archaeologists specialised in Iberian culture rated generated vs. real samples, which scored 3.88 vs. 3.93 on average — ["Reconstruction of Iberian ceramic potteries using generative adversarial networks", Sci. Rep. 2022](https://www.nature.com/articles/s41598-022-14910-7) (figures from search-result snippet; authors not verified)
- **Roman coins.** Evaluators judged coloration, edges, lettering and cracks, plus a real-vs-generated test with 5 judges — [Altaweel, Khelifi, Zafar 2024, JCAA](https://doi.org/10.5334/jcaa.146) (protocol details from search snippet)
- **Harappan pigment colour restoration.** A seven-reviewer panel (four field archaeologists) gave an overall composite rating of 4.47 — [ScienceDirect](https://sciencedirect.com/science/article/pii/S2212054826000561) (from search snippet; authors not verified)
- **"Architecture inside the mirage".** Two architectural historians independently scored 600 images from five T2I platforms. Mean accuracy was 42%, ranging from 32% to 52% — [Magrill et al. 2026, arXiv 2601.09169](https://arxiv.org/abs/2601.09169)
- **Multi-agent Chinese architecture.** 20 experts, 20 students and 20 general users rated images on a 5-point Likert scale. There was no layout metric — [PMC12494968](https://pmc.ncbi.nlm.nih.gov/articles/PMC12494968/)

### Inferences
- A CV-based, plan-referenced faithfulness metric (detector or segmenter IoU against the LLM-produced layout) appears to be an *actual gap in heritage evaluation practice*. It is the project's most defensible novelty element, especially because the 2026 Heritage review explicitly criticises metrics "focused on visual plausibility" — [Lombardi et al. 2026](https://doi.org/10.3390/heritage9070246)
- The project's expert-rating component will be compared with HARF's panels of more than 30 people. A small course-project panel should be framed as a pilot.

### Gaps
- I could not confirm whether HARF's "geometric pre-evaluation" uses automated CV (keypoints or segmentation) or manual measurement; the full text was unavailable.
- The Huizhou chapter's "structural fidelity" metric is unspecified in its abstract.

## Are there datasets pairing archaeological text with plans or images (excavation reports + plan drawings, museum catalog text + photos, Hadashot Arkheologiyot)?

### Takeaway
I found no public ML-ready dataset pairing excavation-report text with plan drawings. Hadashot Arkheologiyot is an open, bilingual corpus that includes plans and photos, but no ML or NLP dataset built from it was found. The nearest structured resources are:
- A historical floor-plan dataset with wall masks (Versailles-FP).
- Modern text-plus-floor-plan data (Tell2Design).
- Script-plus-layout-plus-image data in theatre (StagePro-V1).
- Large museum text-image datasets (ArtiFact), which do not contain plans.

### Cited Findings
- **Hadashot Arkheologiyot.** The e-journal is described as an open database of archaeological reports "including photographs, maps, plans and pottery figures", fully bilingual Hebrew/English — [AWOL listing](https://ancientworldonline.blogspot.com/2012/12/oprn-access-journal-hadashot.html); [JSTOR journal page](https://www.jstor.org/journal/hadaarkhexcasurv). The only derived dataset found is a statistical settlement sample (306–1000 CE) built from the journal's first six volumes, which is not an NLP corpus — [Pandora dataset](https://pandoradata.earth/en/dataset/late-andtiquity-early-islamic-settlements-in-israel-and-palestine)
- **Versailles-FP.** Wall-groundtruthed images of Versailles Palace floor plans from the 17th–18th centuries, used for U-Net wall detection — [Swaileh et al. 2021, arXiv 2103.08064](https://arxiv.org/abs/2103.08064). This is relevant as a historical-plan segmentation resource for S4, but it has no paired text.
- **Tell2Design.** More than 80k floor plans paired with natural-language instructions, all modern residential — [arXiv 2311.15941](https://arxiv.org/abs/2311.15941)
- **StagePro-V1.** 276 stage scenes "spanning different historical styles", annotated with scripts, images and 3D layouts — [arXiv 2503.02595](https://arxiv.org/abs/2503.02595)
- **Report-to-structured-data work.** One workshop paper converts excavation documentation (stratigraphic tables, field notes, inventories) into CIDOC-CRM RDF, which is text to triples, not plans — [CEUR-WS Vol-4099](https://ceur-ws.org/Vol-4099/ER25_DC_hariri.pdf). A 2026 arXiv paper builds FAIR knowledge graphs from flat cultural-heritage records — [arXiv 2608.23263](https://arxiv.org/abs/2608.23263) (title only)
- **Museum text-image.** "ArtiFact: A Large-Scale Multi-Modal Cultural Heritage Dataset" (arXiv 2606.09648) appeared in the museum + text-to-image query; I did not read its abstract — [arXiv 2606.09648](https://arxiv.org/abs/2606.09648)

### Inferences
- Building a small aligned corpus would itself be a citable contribution: Hadashot Arkheologiyot report text, plus its plan figures, plus LLM-extracted layouts. Nothing equivalent was found.
- Because no labelled archaeological-plan detector exists, S4 will probably need either a custom segmenter (Versailles-FP-style wall segmentation is a precedent) or open-vocabulary detectors. Their reliability on archaeological categories must be validated.

### Gaps
- A search-engine summary mentioned a Dutch NER corpus built on about 60,000 excavation reports. I could not tie that claim to a specific URL, so it is unverified.
- I did not query the Archaeology Data Service (ADS), tDAR or the IAA archive directly.

## Verdict: does prior art make the novelty claim false?

### Takeaway
**Verdict: partial overlap found.**
- No published or preprinted work found implements the full pipeline on archaeological or heritage data: excavation/heritage text -> LLM structured spatial layout -> layout-conditioned diffusion -> CV-based layout-faithfulness evaluation.
- **Confidence: moderate-high (about 75%)** that no such four-stage heritage work exists in indexed English-language literature as of 2026-11-14.
- **Confidence: high (about 90%)** that the *pipeline as a method* is not novel. It exists in the generic domain (LayoutGPT, VPGen/VPEval, LMD).
- The claim is defensible only if phrased as: first implemented and quantitatively evaluated application of this pipeline to archaeological excavation reports and site-plan layouts, with a CV-based, plan-referenced faithfulness metric.

### Cited Findings
- **Generic four-stage precedent.** LayoutGPT uses LLM layouts, GLIGEN generation and GLIP-based accuracy — [arXiv 2305.15393](https://arxiv.org/html/2305.15393v2). VPGen/VPEval uses an LM for layout, then generation, then expert-module evaluation — [arXiv 2305.15328](https://arxiv.org/abs/2305.15328)
- **Heritage conceptual precedent for S1–S3.** An LLM with retrieval reads the site plan and literature, infers structure, and a generative model renders plan views, elevations and renderings. It names no model and reports no CV evaluation — [Li et al. 2025, ISPRS](https://isprs-archives.copernicus.org/articles/XLVIII-M-9-2025/843/2025/isprs-archives-XLVIII-M-9-2025-843-2025.pdf)
- **Heritage knowledge-to-ControlNet precedent.** Knowledge-graph reasoning guides ControlNet in an MMDiT for heritage facades — [Guo et al. 2026](https://doi.org/10.1007/978-981-92-0615-5_35)
- **Heritage evaluation precedent.** Evidence-structured prompting with geometric tolerance screening and expert panels of more than 30 people — [HARF, DAACH 2026](https://doi.org/10.1016/j.daach.2026.e00554)
- **Heritage-labelled text -> layout -> image -> OCR check.** Calligraphy pages — [CalliMaster, arXiv 2603.12482](https://arxiv.org/abs/2603.12482)
- **Field-level gap statement.** Only 13 of more than 8,700 3D heritage reconstruction papers addressed generative-AI reconstruction, and their evaluation was focused on visual plausibility — [Lombardi et al. 2026](https://doi.org/10.3390/heritage9070246)

### Inferences
- **Recommended novelty phrasing.**
  - Do not write "first text-to-layout-to-image pipeline".
  - Write instead: "to our knowledge, the first implemented pipeline that converts archaeological excavation-report text into LLM-generated site-plan/box layouts, renders them with layout-conditioned diffusion (ControlNet/GLIGEN), and quantitatively measures layout faithfulness with CV models. Prior heritage work either stops at text prompts (HARF; Lu et al. 2025), uses knowledge graphs or procedural line drawings rather than LLM-derived layouts (Guo et al. 2026; Xu et al. 2024), describes such a pipeline only conceptually without CV evaluation (Li et al. 2025), or targets calligraphy page layout (CalliMaster)."
- **Must-cite list.**
  - Li et al. 2025 (Gonghoulong)
  - HARF (DAACH 2026, Heritage 2026)
  - Guo et al. 2026 (Huizhou)
  - CalliMaster
  - Lu et al. 2025
  - Artal-Sanz et al. 2026 (Saint Engracia)
  - Lombardi et al. 2026 (review)
  - LayoutGPT, LMD and VPGen/VPEval
  - Koch et al. 2025 (ControlNet vs. GLIGEN)
  - StageDesigner
  - Tell2Design / ChatHouseDiffusion
  - LayoutBench or C-Bench (for S4 methodology)

### Gaps
Residual risks that could still falsify the claim:
1. Chinese-language literature (CNKI/Wanfang) on 考古 (archaeology) + ControlNet + 大模型 (large models) was not searchable here.
2. Paywalled full texts: the Huizhou chapter, HARF and the Brunel thesis may describe an LLM-to-mask step not visible in their abstracts.
3. Unindexed theses and course projects (ProQuest, university repositories), and student GitHub repos not matching my keywords.
4. Workshop proceedings not enumerated: e-Heritage, VISART, AI4CH, GCH, CAA 2025–2026.
5. Preprints posted in the last few weeks before 2026-11-14 that OpenAlex/arXiv indexing might lag on.
6. Google Scholar was not directly queried.

A final manual check is advisable on Google Scholar (queries: "excavation report" "ControlNet"; "archaeological" "GLIGEN"; "site plan" "layout-to-image" heritage) and on CNKI (考古 ControlNet 布局, i.e. "archaeology ControlNet layout"; 遗址复原 大语言模型 扩散, i.e. "site reconstruction LLM diffusion").
