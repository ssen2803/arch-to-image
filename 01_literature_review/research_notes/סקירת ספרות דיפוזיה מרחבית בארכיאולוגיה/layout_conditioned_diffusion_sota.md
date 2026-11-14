# Text-to-Layout, Layout-Conditioned Diffusion and Spatial-Fidelity Evaluation: General-Domain State of the Art (2022 to Nov 2026)

Scope: stages 2-4 of the planned pipeline (text -> LLM layout -> layout-conditioned diffusion -> CV evaluation). Research date 2026-11-14.

**Verification legend (applies to every bullet):**
- **[V]**: verified this session by opening the primary page (arXiv abstract/HTML, NeurIPS/CVF page) or from search results quoting it.
- **[K]**: taken from the primary paper as I know it from prior reading. The URL is the paper's own arXiv page, but I did not reopen it this session. Re-check exact numbers before quoting them in the report.
- **[L]**: title, venue and arXiv ID come only from the curated list *Awesome-Controllable-T2I-Diffusion-Models* (maintained by the authors of the TPAMI survey). I did not open the paper itself.
- **NEW** marks items from 2025-2026. **NEWEST** marks items from roughly mid-2026 onward.

---

## Q1. Which works implement "text -> (LLM) layout -> layout-conditioned diffusion", and which layout representation does each use? (Plus general text-to-layout methods)

### Takeaway
The two-stage "LLM plans, diffusion renders" pattern was established in 2023: LayoutGPT, LMD, LayoutLLM-T2I, VPGen, Control-GPT, SceneGenie and Attention-Refocusing all plan layouts and then render them, mostly with GLIGEN or a training-free box-guidance controller on SD1.x. In 2024 the pattern grew self-correction and agent loops (SLD, RPG, RealCompo, GenArtist, Ranni). In 2025-2026 it moved into unified multimodal models that reason, plan and paint in one network (PlanGen, GoT, ATLAS "Think, Plan, Paint"). Axis-aligned bounding boxes with a phrase per box remain by far the most common interface. Alternatives are masks/blobs, scene graphs, "semantic panels", TikZ/SVG/vector code, and layouts serialised as CSS/HTML text. The closest structural analogue to the archaeology project is a Nov 2025 two-stage system with a domain-specific fine-tune (table settings). It reports object recall rising from 57.2% to 99.9% on complex scenes.

### Cited Findings

**A. Canonical 2023 two-stage pipelines**
- **LayoutGPT** (Feng et al., NeurIPS 2023, arXiv 2305.15393). GPT-3.5/4 is prompted with in-context exemplars and a CSS-style "style sheet" serialisation, and it outputs bounding boxes for 2D images and 3D indoor scenes. GLIGEN renders the 2D images. The paper introduces the NSR-1K benchmark for numerical and spatial reasoning and scores layouts with layout-level counting/spatial accuracy plus image-level detection with GLIP. Code: github.com/weixi-feng/LayoutGPT. [K] — [arXiv](https://arxiv.org/abs/2305.15393)
- **LMD: LLM-grounded Diffusion** (Lian, Li, Yala, Darrell, TMLR 2024, arXiv 2305.13655). An LLM (GPT-4) outputs captioned boxes plus a background prompt. Stage 2 is a training-free layout-grounded controller on Stable Diffusion; the LMD+ variant uses GLIGEN adapters. Its benchmark has 400 prompts covering negation, generative numeracy, attribute binding and spatial relations, scored with an open-vocabulary detector (OWL-ViT). Code: github.com/TonyLianLong/LLM-groundedDiffusion, also available as a diffusers community pipeline. [K] — [arXiv](https://arxiv.org/abs/2305.13655)
- **LayoutLLM-T2I** (Qu et al., ACM MM 2023, arXiv 2308.05095). ChatGPT generates layouts, with feedback-based selection of in-context examples. A relation-aware, layout-conditioned diffusion model (SD-based) then renders boxes and phrases. Code released on GitHub (LayoutLLM-T2I). [K] — [arXiv](https://arxiv.org/abs/2308.05095)
- **VPGen + VPEval** (Cho, Zala, Bansal, NeurIPS 2023, arXiv 2305.15328). VPGen fine-tunes Vicuna-13B to emit object, count and box layouts step by step; GLIGEN renders them. VPEval is the evaluation half: "visual programs" call expert modules (open-vocabulary detection, OCR, VQA, depth) to score count, spatial relations and scale interpretably. Code: github.com/j-min/VPGen. [K] — [arXiv](https://arxiv.org/abs/2305.15328)
- **Control-GPT** ("Controllable Text-to-Image Generation with GPT-4", Zhang et al., 2023, arXiv 2305.18583). GPT-4 writes **TikZ code** sketches that are rasterised and passed with grounding tokens to a ControlNet. The paper reports better spatial-relation accuracy. I know of no official public code. [K] — [arXiv](https://arxiv.org/abs/2305.18583)
- **SceneGenie** (Farshad et al., ICCV 2023 Workshops, arXiv 2304.14573). Representation is a **scene graph**, from which boxes and segmentation are predicted. Inference-time guidance on SD uses CLIP-based box/segmentation guidance; evaluated on Visual Genome and COCO-Stuff. [K] — [arXiv](https://arxiv.org/abs/2304.14573)
- **Attention Refocusing** ("Grounded Text-to-Image Synthesis with Attention Refocusing", Phung, Ge, Huang, CVPR 2024, arXiv 2306.05427). An LLM (GPT-4) generates boxes from the prompt. Training-free cross-attention and self-attention refocusing losses then enforce them, either on SD or on top of GLIGEN. Evaluated on HRS-Bench and DrawBench (spatial, counting, colour). Code is public. [K] — [arXiv](https://arxiv.org/abs/2306.05427)
- **Reason out Your Layout** (Chen et al., 2023, arXiv 2311.17126). The LLM generates the layout, and a layout-aware adapter injects it into diffusion. [V title] — [arXiv](https://arxiv.org/pdf/2311.17126)
- **Check, Locate, Rectify** (2023, arXiv 2311.15773). A training-free layout-calibration system: it detects layout errors in the generated image and rectifies them. [V title] — [arXiv](https://arxiv.org/pdf/2311.15773)
- **LLM Blueprint** (Gani et al., ICLR 2024, arXiv 2310.10640). The LLM extracts boxes, per-object descriptions and a background from long prompts. Rendering is box-guided with iterative refinement. Code is public. [K] — [arXiv](https://arxiv.org/abs/2310.10640)

**B. 2024: self-correction, balancing, agents**
- **SLD: Self-correcting LLM-controlled Diffusion** (Wu et al., CVPR 2024, arXiv 2311.16090). An LLM and an open-vocabulary detector compare the generated image with the prompt, then issue latent-space edit operations (add, move, delete, attribute change) in a closed loop. Training-free; code: github.com/tsunghan-wu/SLD. [K] — [arXiv](https://arxiv.org/abs/2311.16090)
- **Ranni** (Feng et al., CVPR 2024, arXiv 2311.17002). An LLM converts text into a **"semantic panel"** (per-object box, colour, keypoints, description), and the diffusion model is conditioned on that panel. The panel also supports interactive editing. Code: github.com/ali-vilab/Ranni. [K] — [arXiv](https://arxiv.org/abs/2311.17002)
- **RPG** ("Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs", Yang et al., ICML 2024, arXiv 2401.11708). An MLLM splits the image into complementary rectangular regions with sub-prompts, and training-free regional diffusion renders them. Works with SDXL; code: github.com/YangLing0818/RPG-DiffusionMaster. [K] — [arXiv](https://arxiv.org/abs/2401.11708)
- **RealCompo** (Zhang et al., NeurIPS 2024, arXiv 2402.12908). An LLM produces the layout. A training-free "balancer" mixes a T2I model (realism) with a spatial-aware model (compositionality: GLIGEN/InstanceDiffusion boxes, keypoints or segmentation). Evaluated on T2I-CompBench. Code: github.com/YangLing0818/RealCompo. [K] — [arXiv](https://arxiv.org/abs/2402.12908)
- **GenArtist** (Wang et al., NeurIPS 2024, arXiv 2407.05600). An MLLM agent decomposes the task into a planning tree and calls generation, layout and editing tools. It checks results with detection and self-corrects. [K] — [arXiv](https://arxiv.org/abs/2407.05600)
- **SLayR: Scene Layout Generation with Rectified Flow** (Braunstein et al., arXiv 2412.05003, v2 Mar 2025). A dedicated **non-LLM text-to-layout transformer** trained with rectified flow, designed to feed existing layout-to-image models. It claims to beat baselines, including LLMs, on plausibility and variety while being at least 3x smaller. No code link on the arXiv page. [V] — [arXiv](https://arxiv.org/abs/2412.05003)

**C. 2025-2026 successors (NEW)**
- **PlanGen** (arXiv 2503.10127, ICCV 2025 per citing papers). Unifies layout planning and image generation inside **one autoregressive VLM**. NEW. [V title/ID; venue from a citing reference list] — [arXiv](https://arxiv.org/pdf/2503.10127)
- **GoT: Generation Chain-of-Thought** (Fang et al., 2025, arXiv 2503.10639). An MLLM writes a reasoning chain containing explicit object boxes, then a diffusion renderer (SDXL-based) generates the image. NEW. [K] — [arXiv](https://arxiv.org/abs/2503.10639)
- **A Two-Stage System for Layout-Controlled Image Generation using LLMs and Diffusion Models** (Koch, Krumme, Gadzicki, arXiv 2511.06888, Nov 2025). An LLM turns an object list into a structured layout. To keep it reliable, the LLM generates only core objects and rule-based insertion completes the rest. **ControlNet vs GLIGEN** is compared after **domain-specific fine-tuning on table-setting data**. Object recall on complex scenes rises from 57.2% to 99.9%. Trade-off: ControlNet keeps text/style control but hallucinates objects; GLIGEN has better layout fidelity but weaker prompt control. No code link. NEW. [V] — [arXiv](https://arxiv.org/abs/2511.06888)
- **ATLAS, "Think, Plan, Paint: Layout-Aware Reasoning for Controllable Image Generation in Unified Models"** (Liu et al., arXiv 2607.16409, Jul 2026; listed as NeurIPS 2026 accepted). An MLLM reasons, plans an explicit layout and renders. Layout is the shared interface across the three stages, and an RL-based layout-alignment reward ties image to plan. Built at 7B and 80B scales. Claims +65.31% average over layout-based unified MLLMs, +23.06% on spatial tasks, and a new ATLAS-Reasoning benchmark. No code link. NEWEST. [V; venue L] — [arXiv](https://arxiv.org/abs/2607.16409)
- Other 2026 LLM/agentic layout-planning entries, NEWEST, all [L]:
  - SketchDirector (agentic layout reasoning for complex scene sketching, arXiv 2610.41260)
  - SpatialGuard (verifiable spatial reasoning for T2I, arXiv 2609.01582)
  - Agentic Flow Steering and Parallel Rollout Search for Spatially Grounded T2I (arXiv 2603.18627)
  - EchoGen, unified layout-image generation and understanding with cycle consistency (AAAI 2026, arXiv 2603.18001)
  - DuetGen, joint AR-layout plus diffusion for visual text (arXiv 2609.22916) [V title/abstract snippet]

  — [Awesome list README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md); [DuetGen arXiv](https://arxiv.org/abs/2609.22916)

**D. General text-to-layout / layout-generation methods (non-photographic, useful as stage-2 baselines)**
- **LayoutTransformer** (Gupta et al., ICCV 2021, arXiv 2006.14615). Autoregressive layout generation. [K] — [arXiv](https://arxiv.org/abs/2006.14615)
- **LayoutDM** (Inoue et al., CVPR 2023, arXiv 2303.08137). Discrete diffusion for graphic layouts. [K] — [arXiv](https://arxiv.org/abs/2303.08137)
- **LayoutPrompter** (Lin et al., NeurIPS 2023, arXiv 2311.06495). Training-free LLM in-context layout generation with exemplar retrieval and ranking. [K] — [arXiv](https://arxiv.org/abs/2311.06495)
- **LayoutNUWA** (Tang et al., ICLR 2024, arXiv 2309.09506). Treats layout generation as **HTML code** completion with an LLM. [K] — [arXiv](https://arxiv.org/abs/2309.09506)
- **Tell2Design** (Leng et al., ACL 2023, arXiv 2311.15941). Language-to-**floor-plan** dataset (about 80k plans with natural-language instructions) plus a Seq2Seq baseline. The most direct "report text -> plan" analogue in the architecture domain. [K] — [arXiv](https://arxiv.org/abs/2311.15941)
- **HouseDiffusion** (Shabani et al., CVPR 2023, arXiv 2211.13287). Diffusion over vector floor-plan polygons, conditioned on a room-adjacency (bubble) graph. [K] — [arXiv](https://arxiv.org/abs/2211.13287)
- 3D/scene variants:
  - Holodeck (Yang et al., CVPR 2024, arXiv 2312.09067): LLM-generated 3D environments
  - LayoutVLM (Sun et al., CVPR 2025, arXiv 2412.02193): VLM 3D layout optimisation

  [K] — [Holodeck](https://arxiv.org/abs/2312.09067); [LayoutVLM](https://arxiv.org/abs/2412.02193)

### Inferences
- **Representation choice.** Every mainstream stage-3 renderer (GLIGEN, MIGC, InstanceDiffusion, CreatiLayout, InstanceAssemble, 3DIS) accepts **boxes plus a phrase per box**, so an LLM output schema of `[{label, description, bbox:[x0,y0,x1,y1]}]` (LMD/LayoutGPT style) is the path of least resistance. A plan rendered as **segmentation masks** (walls, floors, pits) opens up ControlNet-seg, InstanceDiffusion masks and MIGC++. Code-like representations (TikZ in Control-GPT, HTML in LayoutNUWA, SVG in 2026 work) match the vector/plan nature of excavation drawings and are worth citing as alternatives.
- **Closest analogue.** Koch et al. (2511.06888) is the nearest analogue to the project: narrow domain, LLM plus rule-based layout completion, GLIGEN vs ControlNet, recall-based evaluation. It is a natural "related work / baseline design" anchor.
- **Positioning.** The 2025-2026 trend toward unified MLLMs (PlanGen, GoT, ATLAS) is too compute-heavy for a course project. They are best cited as SOTA context rather than used as baselines.

### Gaps
- I did not re-verify which of LayoutLLM-T2I, VPGen and Control-GPT have public **weights** today (repos may be stale). Control-GPT code availability is unconfirmed.
- PlanGen's venue (ICCV 2025) comes from a citing paper's reference list, not the official proceedings.
- SketchDirector, SpatialGuard and Agentic Flow Steering were seen only as list entries. I did not read their methods.
- I found no general-domain work that goes from long narrative technical reports (as opposed to short prompts) to layouts. LLM Blueprint (long prompts) is the nearest.

---

## Q2. Which spatial conditioning mechanisms are dominant and practical on a student GPU budget, and which work with SDXL / SD3 / FLUX?

### Takeaway
There are three families:
1. **Dense map control**: ControlNet, T2I-Adapter, ControlNet-XS, ControlNet++, Uni-ControlNet. They take segmentation, scribble, lineart, MLSD, depth or edges.
2. **Trained box/instance adapters**: GLIGEN, LayoutDiffusion, InstanceDiffusion, MIGC/MIGC++, HiCo, ROICtrl, and on DiTs CreatiLayout, EliGen and InstanceAssemble.
3. **Training-free attention/latent guidance**: MultiDiffusion, Layout-guidance, BoxDiff, Attention-Refocusing, DenseDiffusion, and on DiTs Regional Prompting-FLUX, 3DIS-FLUX and DreamRenderer.

For a student budget, SD1.x/SDXL with GLIGEN, MIGC, InstanceDiffusion or ControlNet-seg/lineart/MLSD is the most practical: public weights, diffusers support, fits on 12-24 GB. FLUX/SD3 control (FLUX.1 Canny/Depth, EasyControl, OminiControl, 3DIS-FLUX, InstanceAssemble LoRA, CreatiLayout) is SOTA but needs about 24 GB or quantisation.

### Cited Findings

**Dense / map-based control (segmentation, scribble, lineart, MLSD, depth)**
- **ControlNet** (Zhang, Rao, Agrawala, ICCV 2023, arXiv 2302.05543). A trainable copy of the SD encoder is attached via zero-initialised convolutions. Released conditions include Canny, depth, HED/soft-edge, **MLSD straight lines**, normals, OpenPose, **scribble**, **ADE20K segmentation**, with lineart added in v1.1. The authors state training is as fast as fine-tuning and feasible on personal GPUs. Code and weights: github.com/lllyasviel/ControlNet; SD1.5 v1.1 weights on Hugging Face. [K] — [arXiv](https://arxiv.org/abs/2302.05543)
- **T2I-Adapter** (Mou et al., AAAI 2024, arXiv 2302.08453). A lightweight adapter (tens of millions of parameters) with a frozen base. SDXL adapters (sketch, canny, lineart, depth, openpose) were released by TencentARC. [K] — [arXiv](https://arxiv.org/abs/2302.08453)
- **Uni-ControlNet** (Zhao et al., NeurIPS 2023, arXiv 2305.16322) and **UniControl** (Qin et al., NeurIPS 2023, arXiv 2305.11147) combine multiple conditions in one model. [K] — [Uni-ControlNet](https://arxiv.org/abs/2305.16322); [UniControl](https://arxiv.org/abs/2305.11147)
- **ControlNet-XS** (Zavadski et al., ECCV 2024, arXiv 2312.06573). Uses a small fraction of ControlNet's parameters with high-bandwidth feedback between branches. SD1.5/SD2.1/SDXL variants. [K] — [arXiv](https://arxiv.org/abs/2312.06573)
- **ControlNet++** (Li et al., ECCV 2024, arXiv 2404.07987). Adds pixel-level cycle-consistency rewards: a pretrained discriminative model (segmenter, depth estimator, edge detector) re-extracts the condition from the generated image. Abstract-level gains over ControlNet: about +11.1% mIoU (segmentation), +13.4% SSIM (lineart), +7.6% RMSE (depth). This paper also defines the standard **condition-fidelity metrics** (see Q3). Code public. [K] — [arXiv](https://arxiv.org/abs/2404.07987)
- **ALDM: "Adversarial Supervision Makes Layout-to-Image Diffusion Models Thrive"** (Li et al., ICLR 2024, arXiv 2401.08815). Uses a segmenter as adversarial or feedback supervision on a ControlNet-style model to improve **mIoU alignment with the input segmentation map**. Plan-as-segmentation is a direct analogue. [L for listing; method K] — [arXiv](https://arxiv.org/abs/2401.08815)
- Segmentation-to-image (L2I from masks):
  - DenseDiffusion (Kim et al., ICCV 2023, arXiv 2308.12964): training-free attention modulation from segment-plus-text
  - ZestGuide (Couairon et al., ICCV 2023, arXiv 2306.13754): zero-shot segmentation guidance
  - FreestyleNet (Xue et al., CVPR 2023, arXiv 2303.14412)
  - SpaText (Avrahami et al., CVPR 2023, arXiv 2211.14305)
  - PLACE (CVPR 2024, arXiv 2403.01852)

  [K] — [DenseDiffusion](https://arxiv.org/abs/2308.12964); [ZestGuide](https://arxiv.org/abs/2306.13754); [FreestyleNet](https://arxiv.org/abs/2303.14412); [SpaText](https://arxiv.org/abs/2211.14305); [PLACE](https://arxiv.org/abs/2403.01852)
- Adapter portability: Ctrl-Adapter (ICLR 2025, arXiv 2404.09967) adapts existing ControlNets to new backbones (SDXL, video). CtrLoRA (ICLR 2025, arXiv 2410.09400) is an extensible LoRA-based multi-condition ControlNet. [L] — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)

**Trained box / instance adapters (UNet era, SD1.x)**
- **GLIGEN** (Li et al., CVPR 2023, arXiv 2301.07093). Gated self-attention layers are inserted into a **frozen SD1.4**; grounding tokens are made from box coordinates and phrase embeddings. Also supports keypoints, depth, edge and semantic maps. Open-set box grounding. Code and weights: github.com/gligen/GLIGEN; available as `StableDiffusionGLIGENPipeline` in diffusers. [K] — [arXiv](https://arxiv.org/abs/2301.07093)
- **LayoutDiffusion** (Zheng et al., CVPR 2023, arXiv 2303.17189). A layout-fusion module plus object-aware cross-attention, trained on COCO-Stuff/VG. **Closed-set** (fixed class vocabulary). Metrics: FID, CAS, YOLOScore, diversity. Code public. [K] — [arXiv](https://arxiv.org/abs/2303.17189)
- **InstanceDiffusion** (Wang et al., CVPR 2024, arXiv 2402.03290). Per-instance captions with **points, scribbles, boxes or instance masks** (UniFusion, ScaleU, multi-instance sampler) on SD1.5. Reports large AP gains on COCO measured with YOLOv8 (about +20 AP50-box and +25 mask IoU over prior SOTA; check exact numbers). Code: github.com/frank-xwang/InstanceDiffusion. [K] — [arXiv](https://arxiv.org/abs/2402.03290)
- **MIGC** (Zhou et al., CVPR 2024, arXiv 2402.05408). "Divide and conquer" instance shaders on SD1.4. Introduces the **COCO-MIG** benchmark with Instance Success Rate (ISR) and mIoU. **MIGC++** (arXiv 2407.02329, TPAMI) adds masks and reference images. Code: github.com/limuloo/MIGC. [K] — [MIGC](https://arxiv.org/abs/2402.05408); [MIGC++](https://arxiv.org/abs/2407.02329)
- **HiCo** (NeurIPS 2024, arXiv 2410.14324). A hierarchical multi-branch controllable model with the HiCo-7K benchmark; code github.com/360CVGroup/HiCo_T2I. **ROICtrl** (CVPR 2025, arXiv 2411.17949). ROI-Align/ROI-Unpool instance control, compatible with community fine-tunes. **IFAdapter** (ICCV 2025, arXiv 2409.08240). [L; HiCo code link from list] — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **Rich-context L2I** (Cheng et al., NeurIPS 2024, arXiv 2409.04847). A regional cross-attention module for long per-box descriptions (see Q3 for its evaluation critique). [V] — [arXiv](https://arxiv.org/abs/2409.04847)

**Training-free box/region guidance (zero training cost)**
- **MultiDiffusion** (Bar-Tal et al., ICML 2023, arXiv 2302.08113). Fuses several diffusion paths over region masks or boxes. It is the training-free baseline that 3DIS-FLUX compares against. [K] — [arXiv](https://arxiv.org/abs/2302.08113)
- **Layout-guidance** ("Training-Free Layout Control with Cross-Attention Guidance", Chen, Laina, Vedaldi, WACV 2024, arXiv 2304.03373). Forward and backward guidance on cross-attention maps. [K] — [arXiv](https://arxiv.org/abs/2304.03373)
- **BoxDiff** (Xie et al., ICCV 2023, arXiv 2307.10816). Inner-box, outer-box and corner constraints on cross-attention, on SD1.4 or on top of GLIGEN. Code public. [K] — [arXiv](https://arxiv.org/abs/2307.10816)
- Other training-free box methods:
  - Directed Diffusion (AAAI 2024, arXiv 2302.13153)
  - R&B (arXiv 2310.08872)
  - LoCo (ACM MM 2025, arXiv 2311.12342)
  - Bounded Attention, "Be Yourself" (arXiv 2403.16990)
  - NoiseCollage (arXiv 2403.03485)
  - GrounDiT (NeurIPS 2024, arXiv 2410.20474): noisy patch transplantation on a DiT
  - ToLo (arXiv 2503.01667): high-overlap layouts
  - "Control and Realism: Best of Both Worlds in Layout-to-Image without Training" (ICML 2025, arXiv 2506.15563)
  - Stitch (arXiv 2509.26644): training-free position control in MM-DiT
  - FreLay (AAAI 2026)
  - AnyCanvas (ICML 2026)

  [L except Directed Diffusion K] — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)

**DiT era: SD3 / SD3.5 / FLUX (NEW)**
- **FLUX.1 Tools** (Black Forest Labs, Nov 2024). Official FLUX.1 Canny and Depth (full model and LoRA), Fill and Redux. FLUX.1-dev is a 12B-parameter model, so full-precision inference needs about 24 GB unless quantised. [K] — [BFL announcement](https://blackforestlabs.ai/flux-1-tools/)
- **OminiControl** (Tan et al., arXiv 2411.15098). Minimal-parameter (about 0.1%) image/spatial conditioning for FLUX. **EasyControl** (ICCV 2025, arXiv 2503.07027). Efficient, flexible condition-injection LoRA for DiTs. Other DiT controllers: NanoControl (arXiv 2508.10424), RelaCtrl (arXiv 2502.14377), FlexControl (ICML 2025, arXiv 2502.10451). NEW. [OminiControl K; others L] — [OminiControl](https://arxiv.org/abs/2411.15098); [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **CreatiLayout** (Zhang et al., ICCV 2025, arXiv 2412.03859). SiamLayout, a Siamese MM-DiT branch that treats layout as its own modality, on SD3/SD3.5 (FLUX variant released). Comes with the large **LayoutSAM** dataset and the **LayoutSAM-Eval** benchmark. Code: github.com/HuiZhang0812/CreatiLayout. NEW. [venue and code V via list; architecture details K] — [arXiv](https://arxiv.org/abs/2412.03859)
- **EliGen** (arXiv 2501.01097). Entity-level regional attention for FLUX, trained with a dataset of entity masks. NEW. [L] — [arXiv](https://arxiv.org/abs/2501.01097)
- **Training-free Regional Prompting for DiTs (FLUX)** (arXiv 2411.02395). NEW. [L] — [arXiv](https://arxiv.org/abs/2411.02395)
- **3DIS** (ICLR 2025, arXiv 2410.12669) and **3DIS-FLUX** (arXiv 2501.05131, Jan 2025). Stage 1 generates a scene **depth map** from the layout. Stage 2 renders with **FLUX.1-Depth-dev** while editing FLUX joint-attention masks per instance. Claims +41% ISR over MultiDiffusion, the previous training-free SOTA, and beats adapter methods. Code open. NEW. [V] — [arXiv](https://arxiv.org/abs/2501.05131); [HF post](https://huggingface.co/posts/sanaka87/168588897325374)
- **DreamRenderer** (ICCV 2025, arXiv 2503.12885). Training-free multi-instance **attribute control** for FLUX ("Taming Multi-Instance Attribute Control in Large-Scale T2I Models"). NEW. [L; FLUX base per 3DIS-FLUX citing text] — [arXiv](https://arxiv.org/abs/2503.12885)
- **InstanceAssemble** (Xiang et al., NeurIPS 2025, arXiv 2509.16691). "Instance assembling attention" with box position plus text or visual content, added to **DiT-based T2I models via lightweight LoRA**. SOTA on complex/dense layouts and compatible with style LoRAs. Introduces **DenseLayout** (5k images, 90k instances) and the **Layout Grounding Score (LGS)**. Code and models: github.com/FireRedTeam/InstanceAssemble. Base models (SD3-Medium / FLUX.1-dev) are not stated on the abstract page. NEW. [V] — [arXiv](https://arxiv.org/abs/2509.16691); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115776)
- 2025-2026 DiT layout controllers, all [L], NEW/NEWEST:
  - Laytrol (arXiv 2511.07934): layout control preserving MM-DiT priors
  - Hybrid Layout Control for DiT (ICCV 2025)
  - ContextGen (ICLR 2026, arXiv 2510.11000)
  - I-DRUID (ICLR 2026)
  - InstanceControl (ECCV 2026, arXiv 2606.31924): no instance labels needed
  - Layer-wise Instance Binding for regional and occlusion control in DiTs (CVPR 2026, arXiv 2603.05769)
  - OcclusionFormer (ICML 2026, arXiv 2605.21343)
  - InstancePin (arXiv 2608.00588)
  - BBQ-to-Image, numeric box and colour control in large T2I models (arXiv 2602.20672)
  - MaskAttn-SDXL (arXiv 2509.15357)

  — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **Survey to cite:** Cao, Zhou, Song, Yang, "Controllable Generation with Text-to-Image Diffusion Models: A Survey", **IEEE TPAMI vol. 48 no. 4, pp. 4771-4791, Apr 2026**, DOI 10.1109/TPAMI.2025.3646548, arXiv 2403.04279. [V] — [README / CITATION](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)

### Inferences
- **Practical student stack (inference only, 12-16 GB):**
  - GLIGEN in diffusers (boxes + phrases, SD1.4)
  - MIGC or InstanceDiffusion (SD1.x; box/mask instance control)
  - ControlNet v1.1 seg/lineart/MLSD/scribble (SD1.5)
  - Training-free BoxDiff / Layout-guidance / Attention-Refocusing as zero-training baselines

  These are the standard baselines in almost every L2I paper, so comparisons are cheap and defensible.
- **Mid-tier (about 24 GB, or 16 GB with fp8/NF4 quantisation):**
  - SDXL ControlNet / T2I-Adapter
  - SD3.5 / FLUX with CreatiLayout, InstanceAssemble-LoRA, 3DIS-FLUX or Regional-Prompting-FLUX

  This is the "modern" comparison point. I did not verify per-method VRAM numbers; they follow from model size (FLUX.1-dev is 12B).
- **Fine-tuning.** Training a ControlNet on a small domain set (for example site-plan segmentation paired with photos) is feasible on one consumer GPU at SD1.5 resolution, per the ControlNet authors' claim. GLIGEN-style or InstanceDiffusion retraining is heavier. LoRA-based InstanceAssemble/EasyControl are the cheapest way to adapt DiTs.
- **Archaeological plans** are line drawings: walls as straight segments, contours, hatching. MLSD/lineart/scribble ControlNets and segmentation ControlNets map more naturally onto them than box-only controllers. A hybrid (segmentation map for structures plus boxes for small finds) mirrors InstanceDiffusion and MIGC++ multi-format inputs.

### Gaps
- Exact VRAM/throughput per method was not verified. Treat GPU-budget statements as inference from parameter counts.
- Exact base models and code status of many 2026 list entries (Laytrol, InstanceControl, I-DRUID, OcclusionFormer, etc.) were not checked.
- The InstanceAssemble base model (SD3 vs FLUX) and the exact LGS definition were not visible on the abstract page.

---

## Q3. How is spatial/layout fidelity measured, and which metrics are most reliable?

### Takeaway
Layout fidelity is measured in four ways:
1. **Detector-based localisation against input boxes**: YOLOv4/v8 AP (GLIGEN, InstanceDiffusion), GroundingDINO ISR/mIoU (MIGC/COCO-MIG), OWL-ViT/OWLv2 IoU-accuracy (LMD, VISOR, the 2026 C-Bench/O-Bench), GLIP (LayoutGPT), UniDet (T2I-CompBench, HRS-Bench), Mask2Former (GenEval).
2. **Segmentation/condition re-extraction**: mIoU, SSIM, F1, RMSE (ControlNet++, ALDM).
3. **VQA/VLM judges** for semantics: TIFA, DSG, VQAScore, Soft-TIFA, VLM-based LayoutSAM-Eval.
4. **Distributional quality**: FID, KID, CMMD.

The most reliable spatial signal is still an **open-vocabulary detector plus IoU matching against the input layout**, validated against humans. A 2026 CVPR Findings benchmark shows OWLv2+IoU "closely aligned" with human annotation on 448 images. Closed-vocabulary COCO detectors drift and fail out of domain: GenEval 2 (Dec 2025) reports up to 17.7% absolute error versus humans. CLIPScore and FID are weak or unreliable for spatial faithfulness.

### Cited Findings

**Layout-specific benchmarks and protocols**
- **LayoutBench** ("Diagnostic Benchmark and Iterative Inpainting for Layout-Guided Image Generation", Cho, Li, Yang, Bansal et al., CVPR 2024 Workshops (EvGenFM), arXiv 2304.06671). Diagnoses four spatial skills (**number, position, size, shape**) on CLEVR-style out-of-distribution layouts, and reports **layout accuracy as detector AP**. Also proposes the IterInpaint baseline. [V] — [arXiv](https://arxiv.org/abs/2304.06671); [CVF](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Cho_Diagnostic_Benchmark_and_Iterative_Inpainting_for_Layout-Guided_Image_Generation_CVPRW_2024_paper.pdf)
- **GLIGEN protocol.** COCO2014-val boxes are fed in, and a pretrained **YOLOv4** detects objects in the generated images; AP/AP50/AP75 are reported against the input boxes, along with FID. LVIS uses a GLIP detector for zero-shot classes. This is the de facto "YOLO score" lineage, together with LayoutDiffusion's YOLOScore. [K] — [GLIGEN](https://arxiv.org/abs/2301.07093); [LayoutDiffusion](https://arxiv.org/abs/2303.17189)
- **InstanceDiffusion protocol.** **YOLOv8** box/mask AP on COCO, with separate metrics for point and scribble adherence. [K] — [arXiv](https://arxiv.org/abs/2402.03290)
- **COCO-MIG (MIGC).** Prompts have 2-6 instances with assigned attributes. **Instance Success Rate** counts an instance as successful if it is detected inside its box at an IoU threshold with the correct attribute, using GroundingDINO. mIoU is also reported. 3DIS and 3DIS-FLUX report ISR as their headline metric. [K for MIGC protocol; ISR use in 3DIS-FLUX V] — [MIGC](https://arxiv.org/abs/2402.05408); [3DIS-FLUX](https://arxiv.org/abs/2501.05131)
- **NEWEST: C-Bench / O-Bench** (Parolari, Faccioli, Ballan, "Benchmarking Layout-Guided Diffusion Models through Unified Semantic-Spatial Evaluation in Closed and Open Settings", **CVPR 2026 Findings**, arXiv 2604.25358). 319,086 generated images.
  - Layout score: the **OWLv2** top-confidence detection per object is matched to the ground-truth box by IoU, Accuracy@IoU is computed at thresholds 0-1 in steps of 0.1, and the score is the **area under the Acc@k curve**.
  - Text score: **TIFA** with BLIP-2 VQA (Llama-2 questions, UnifiedQA filtering).
  - Human validation: TIFA checked on 640 images, good agreement except colour-binding cases. The detector-based score was checked on 448 manually annotated images and was "closely aligned"; humans were slightly more precise at localisation.
  - Models: MIGC, GLIGEN, BoxDiff, Attention-Refocusing, Cross-Attention-Guidance, plus SD as a text-only baseline. **MIGC is best** (unified 0.7082 C-Bench / 0.7548 O-Bench). Performance falls for all models as object count rises. Cross-Attention Guidance has good text alignment but poor layout.
  - Code: github.com/lparolari/cobench.

  [V] — [arXiv HTML](https://arxiv.org/html/2604.25358); [arXiv abs](https://arxiv.org/abs/2604.25358)
- **NEW: 7Bench** (Izzo, Parolari, Vezzaro, Ballan, ICIAP 2025, arXiv 2508.12919). Layout-guided T2I benchmark with seven scenarios covering object generation, colour fidelity, attribute recognition, inter-object relations and spatial control. It defines a **layout alignment score** alongside text alignment. Code: github.com/Elizzo/7Bench. [V; detector not named on abstract page] — [arXiv](https://arxiv.org/abs/2508.12919)
- **NEW: DenseLayout + Layout Grounding Score** (InstanceAssemble, NeurIPS 2025). 5k images, 90k instances. Targets **dense layouts**, which is directly relevant to crowded archaeological scenes. [V] — [arXiv](https://arxiv.org/abs/2509.16691)
- **Rich-context L2I** (NeurIPS 2024). Argues that open-vocabulary L2I methods are trained open-set but **evaluated in closed-set environments**. Proposes two open-vocabulary metrics validated by a user study. [V] — [arXiv](https://arxiv.org/abs/2409.04847)
- **LayoutSAM-Eval** (CreatiLayout, ICCV 2025). A VLM judges per-entity spatial and attribute correctness inside each box. [K] — [arXiv](https://arxiv.org/abs/2412.03859)
- **VPEval** (NeurIPS 2023). Interpretable, module-based (detector, OCR, VQA, depth) checks of count, spatial and scale skills. [K] — [arXiv](https://arxiv.org/abs/2305.15328)

**General compositional T2I benchmarks with spatial/counting splits**
- **T2I-CompBench** (Huang et al., NeurIPS 2023 Datasets & Benchmarks, arXiv 2307.06350). 6k prompts over colour, shape, texture, **spatial**, non-spatial and complex. Spatial is scored with the **UniDet** detector plus a rule on box centres. Attributes use BLIP-VQA, non-spatial uses CLIPScore, complex uses a 3-in-1 combination. **T2I-CompBench++** (TPAMI 2025) adds 3D-spatial and numeracy. [K] — [arXiv](https://arxiv.org/abs/2307.06350)
- **GenEval** (Ghosh, Hajishirzi, Schmidt, NeurIPS 2023 D&B, arXiv 2310.11513). Tasks: single object, two objects, **counting**, colours, **position**, attribute binding. Uses a COCO-trained Mask2Former detector; position is judged from relative box centres, colour by CLIP on the masked crop. The paper itself notes its detector is limited to MS-COCO classes and that "detectors trained primarily on photos do not generalize well to visually distinct art". [V for limitations; protocol K] — [arXiv](https://arxiv.org/pdf/2310.11513)
- **NEW: GenEval 2** (Kamath et al., arXiv 2512.16853, Dec 2025).
  - GenEval has **drifted from human judgement**, with absolute error up to **17.7%** for current models, and is "effectively saturated".
  - A root cause is COCO co-occurrence bias in the detector; one example is a correctly drawn toothbrush detected as a bird.
  - Proposes the VQA-based **Soft-TIFA**, which it reports drifts less than VQAScore.
  - Top models reach 85.3% atom-level but only 35.8% prompt-level accuracy.

  [V] — [arXiv](https://arxiv.org/pdf/2512.16853)
- **HRS-Bench** (Bakr et al., ICCV 2023, arXiv 2304.05390). 13 skills in 5 categories, including counting and spatial composition, scored with detector-based (UniDet) checks. [K] — [arXiv](https://arxiv.org/abs/2304.05390)
- **VISOR / SR2D** (Gokhale et al., arXiv 2212.10015, "Benchmarking Spatial Relationships in Text-to-Image Generation"). About 25k two-object relation prompts. VISOR scores relation correctness, conditional on both objects being detected (OWL-ViT). The key finding is that models often fail to generate both objects at all. [K] — [arXiv](https://arxiv.org/abs/2212.10015)
- **DrawBench** (Imagen, Saharia et al., NeurIPS 2022, arXiv 2205.11487). 200 prompts in 11 categories, including counting and positional. Scored by **human** pairwise preference. [K] — [arXiv](https://arxiv.org/abs/2205.11487)

**Semantic-alignment scorers**
- **CLIPScore** (Hessel et al., EMNLP 2021, arXiv 2104.08718). CLIP-family models behave like "bags of words" and are weak on relations and order (Yuksekgonul et al., ICLR 2023, arXiv 2210.01936). [K] — [CLIPScore](https://arxiv.org/abs/2104.08718); [Bag-of-words](https://arxiv.org/abs/2210.01936)
- **TIFA** (Hu et al., ICCV 2023, arXiv 2303.11897), **DSG** (Cho et al., ICLR 2024, arXiv 2310.18235) and **VQAScore** (Lin et al., ECCV 2024, arXiv 2404.01291; CLIP-FlanT5, GenAI-Bench). VQA-based scores agree with humans better than CLIPScore on compositional prompts, but they judge semantics, not box-level localisation. [K] — [TIFA](https://arxiv.org/abs/2303.11897); [DSG](https://arxiv.org/abs/2310.18235); [VQAScore](https://arxiv.org/abs/2404.01291)

**Condition-fidelity metrics for dense maps (ControlNet++ protocol)**
- The generated image is re-segmented and **mIoU** computed against the input segmentation (ADE20K/COCO-Stuff, with UperNet/Mask2Former segmenters). Other conditions use **SSIM** for HED/lineart, **F1** for Canny, and **RMSE** for depth (re-estimated depth vs input depth). [K] — [ControlNet++](https://arxiv.org/abs/2404.07987)

**Distributional realism**
- FID (Heusel et al., NeurIPS 2017, arXiv 1706.08500) and KID (Bińkowski et al., ICLR 2018, arXiv 1801.01401; unbiased, better for small sample sizes) are standard. **CMMD** (Jayasumana et al., CVPR 2024, arXiv 2401.09603) argues FID's Inception features and normality assumption misrepresent modern T2I quality and recommends MMD on CLIP embeddings. [K] — [FID](https://arxiv.org/abs/1706.08500); [KID](https://arxiv.org/abs/1801.01401); [CMMD](https://arxiv.org/abs/2401.09603)

### Inferences
- **Reliability ranking for spatial faithfulness:**
  1. Open-vocabulary detector (OWLv2/GroundingDINO) with IoU-based accuracy/AP against the input boxes, plus a human-validation subset (C-Bench/O-Bench protocol).
  2. Segmentation mIoU against the input mask, when the layout is a plan/mask.
  3. Closed-set YOLO AP (comparable to legacy papers, but closed vocabulary).
  4. VQA judges (TIFA/VQAScore) for "is the right thing there" semantics.
  5. CLIPScore and FID, which are not spatial metrics at all.
- **Domain risk.** For archaeology, a COCO-trained detector has no classes for walls, pits, installations or sherds. Its photo/co-occurrence biases (documented in GenEval 2 and in the GenEval paper's own caveats) make off-the-shelf detector scores unreliable. Two fixes:
  - (a) Fine-tune a YOLO/GroundingDINO detector, or a segmenter, on real annotated excavation photos, then compute AP50 / Acc@IoU-AUC / mIoU of generated images against the input layout (the GLIGEN/ControlNet++ protocol transplanted to the domain).
  - (b) Validate that detector against human labels on a few hundred images, as Parolari et al. did with 448 images.
- **Count by instance.** Use per-instance success (ISR-style) and per-class counting accuracy rather than a single mAP, because many-small-object scenes hide failures in averaged AP.
- **Realism.** Report KID or CMMD rather than FID for small domain test sets (a few hundred to a few thousand images).

### Gaps
- I could not confirm the detector used by 7Bench or the exact LGS definition in InstanceAssemble from the abstract pages.
- I did not open the T2I-CompBench++ TPAMI version to confirm its 3D-spatial/numeracy evaluators.
- I found no published human-agreement study for detector-based layout metrics on **non-photographic** (drawing, plan, scientific) imagery.

---

## Q4. Known failure modes (counting, spatial relations, attribute leakage, small objects), and what they imply for archaeological scenes with many small, similar objects

### Takeaway
Every evaluated family degrades as the number of objects grows. Counting and relations fail in plain T2I, which is the very motivation for explicit layouts. Attribute leakage between instances persists even with box control. Small boxes are under-served because cross-attention maps are low-resolution, and dense or overlapping layouts are an active 2025-2026 research front (InstanceAssemble/DenseLayout, ToLo, OcclusionFormer). Archaeological scenes (dozens of small, visually similar walls, installations and pottery) sit at the hardest end of every benchmark axis.

### Cited Findings
- In the 2026 C-Bench/O-Bench study, performance **declines with object count for all models**. Layout-aware models keep an advantage at higher counts. "Complex compositions" is the hardest scenario, with small gaps between models. [V] — [arXiv HTML](https://arxiv.org/html/2604.25358)
- Without layout control, object recall on complex table-setting scenes is only **57.2%**; with LLM layout plus layout-conditioned diffusion it reaches 99.9%. ControlNet **hallucinates objects**, while GLIGEN follows the layout but loses prompt controllability. [V] — [arXiv](https://arxiv.org/abs/2511.06888)
- GenEval 2 (Dec 2025): even top models get only **35.8% prompt-level** accuracy versus 85.3% atom-level on compositional prompts. Errors compound as more constraints appear in one prompt. [V] — [arXiv](https://arxiv.org/pdf/2512.16853)
- LayoutBench explicitly stresses **number, position, size and shape** under out-of-distribution layouts; layout-guided models degrade sharply on unusual sizes and positions. [V for skill set; degradation finding K] — [arXiv](https://arxiv.org/abs/2304.06671)
- VISOR: T2I models frequently fail to generate both objects of a two-object relation prompt, which conflates "object omission" with "relation error". [K] — [arXiv](https://arxiv.org/abs/2212.10015)
- Attribute binding and neglect:
  - Attend-and-Excite (Chefer et al., SIGGRAPH 2023, arXiv 2301.13826) identifies "catastrophic neglect" and attribute mis-binding.
  - Bounded Attention ("Be Yourself", arXiv 2403.16990) attributes **semantic leakage between subjects** to shared attention.
  - DreamRenderer (ICCV 2025) and MIGC target multi-instance attribute control specifically.

  [K/L] — [Attend-and-Excite](https://arxiv.org/abs/2301.13826); [Bounded Attention](https://arxiv.org/abs/2403.16990); [DreamRenderer](https://arxiv.org/abs/2503.12885)
- Counting-specific work: Make It Count / CountGen (CVPR 2025, arXiv 2406.10210) and "Mitigating Noise-Induced Layout Priors for Object Counting in Diffusion Models" (ICML 2026). TokenCompose (CVPR 2024, arXiv 2312.03626) adds token-level segmentation supervision for multi-category composition. [K/L] — [CountGen](https://arxiv.org/abs/2406.10210); [TokenCompose](https://arxiv.org/abs/2312.03626); [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- Dense, overlapping and occluded layouts are explicit 2025-2026 targets:
  - InstanceAssemble (DenseLayout, 90k instances in 5k images; "training-free heuristics often fail in crowded scenes")
  - ToLo (high-overlap layouts)
  - LaRender (ICCV 2025, occlusion)
  - OcclusionFormer (ICML 2026, z-order)
  - Layer-wise Instance Binding (CVPR 2026)
  - OccluRank (arXiv 2608.20932)

  [V InstanceAssemble; others L] — [InstanceAssemble](https://arxiv.org/abs/2509.16691); [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **Realism vs control trade-off:** strong layout control reduces realism and prompt adherence, the motivation for RealCompo (NeurIPS 2024) and "Control and Realism" (ICML 2025). [K/L] — [RealCompo](https://arxiv.org/abs/2402.12908); [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **Detector-side failure** compounds generation failure: COCO-trained detectors misclassify from contextual co-occurrence (GenEval 2's toothbrush-as-bird example), so evaluation can mis-score even a correct generation. [V] — [arXiv](https://arxiv.org/pdf/2512.16853)

### Inferences
- **Small objects.** SD1.x cross-attention runs at 16x16 to 64x64 latent resolution, so a sherd or small installation occupying under 2% of the image may get only a handful of attention cells. Expect box-guided methods to miss or merge such instances. Mitigations:
  - generate at higher resolution (SDXL/FLUX)
  - tile or crop-and-inpaint (IterInpaint-style)
  - represent small finds as point or scribble instances (InstanceDiffusion)
- **Many similar instances** (repeated wall segments, many identical pottery boxes) are the regime where instance leakage and merging dominate. Score with per-instance ISR and count accuracy, not only mAP.
- **Spatial relations** ("north of", "abutting", "cut by") in excavation text are richer than benchmark left/right/above. The LLM-to-layout stage, not the renderer, is where these must be resolved. Evaluate stage 2 separately (layout vs. ground-truth plan IoU) before stage 3.
- **Overlap and stratigraphy** (features cutting other features) map onto the occlusion/z-order line of work, which is still immature.

### Gaps
- I found no benchmark that specifically isolates **tiny-object** fidelity (for example boxes under 32x32 px) in layout-to-image. LayoutBench "size" is closest, but I did not verify its exact size bins.
- There are no quantitative numbers on attribute leakage for visually near-identical instances (same class, slightly different attributes).

---

## Q5. Domain-transfer results: layout-conditioned diffusion on scientific, technical, architectural or non-photographic imagery

### Takeaway
Layout-conditioned diffusion transfers well when the model is fine-tuned on in-domain data and evaluated by a detector trained on that domain. Remote sensing is the most mature transfer domain (AeroGen CVPR 2025, plus many 2026 ECCV/AAAI/NeurIPS works). Driving scenes and detection-data synthesis (GeoDiffusion, ODGEN) establish the protocol "train a detector on synthetic data and measure mAP on real". Floor plans and architecture use ControlNet over SD, but they are evaluated mainly by user studies with no standard geometric metric. 2026 also brings layout control for charts, STEM illustrations, medical images and industrial defects, plus few-shot "atypical" layout-to-image, the most relevant setting for rare archaeological classes.

### Cited Findings
- **AeroGen** (Tang et al., CVPR 2025, arXiv 2411.15497). Layout-controllable diffusion for remote sensing that accepts **horizontal and rotated boxes**, fusing text and layout with dual cross-attention on a fine-tuned LDM. Synthetic data raises detector **mAP by 3.7% (DIOR), 4.3% (DIOR-R) and 2.43% (HRSC)**. Code: github.com/Sonettoo/AeroGen. [V] — [arXiv](https://arxiv.org/html/2411.15497v3); [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_AeroGen_Enhancing_Remote_Sensing_Object_Detection_with_Diffusion-Driven_Data_Generation_CVPR_2025_paper.html)
- **GeoDiffusion** (Chen et al., ICLR 2024, arXiv 2306.04607). Encodes boxes and geometric conditions **as text tokens** for driving-scene generation (nuImages) and evaluates fidelity by training and testing detectors. **ODGEN** (NeurIPS 2024, arXiv 2405.15199) targets domain-specific object-detection data generation. [GeoDiffusion K; ODGEN L] — [GeoDiffusion](https://arxiv.org/abs/2306.04607); [ODGEN](https://arxiv.org/abs/2405.15199)
- **DiffusionSat** (Khanna et al., ICLR 2024, arXiv 2312.03606). A satellite-image diffusion foundation model with metadata and ControlNet-style conditioning. [K] — [arXiv](https://arxiv.org/abs/2312.03606)
- 2026 remote-sensing and geospatial control (NEWEST), all [L]:
  - RSGen, layout-driven RS generation with edge guidance (arXiv 2603.15484)
  - Any2RSI (AAAI 2026)
  - TerraDiT-Ω, unified spatial control with any geospatial primitive (ECCV 2026)
  - SGP2 (ECCV 2026)
  - GeoCore-9B (NeurIPS 2026, arXiv 2608.01896)
  - Geometry-controlled high-resolution satellite synthesis (arXiv 2605.04557)
  - D2-CDIG, DEM plus cloud priors (arXiv 2605.14326)

  — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **Floor plans and architecture:**
  - **WAFFLE** (arXiv 2412.00955) fine-tunes SD on in-the-wild floor-plan images and trains a **ControlNet on CubiCasa5K layouts**. Text-conditioned outputs are evaluated by a **user study** (36 random image pairs, prompts such as "cathedral"). [V via search snippet]
  - **Qiu et al., "Text Semantics to Flexible Design"** (arXiv 2501.09279) uses SD plus a boundary (Canny) ControlNet with room-type and connectivity text, trained on RPLAN (80k+ plans). [V via search snippet/review summary; metrics not seen]
  - A community SD1.5 Canny ControlNet for floor plans (Flanora) warns its outputs are "floor-plan-like" but not necessarily architecturally valid. [V]
  - Panoramic room generation (arXiv 2310.03602) generates a 3D layout first, then renders with ControlNet. [V snippet]

  — [WAFFLE](https://arxiv.org/pdf/2412.00955); [Qiu et al.](https://arxiv.org/pdf/2501.09279); [Flanora model card](https://huggingface.co/BJyotibrat/Flanora-AI-v1); [2310.03602](https://arxiv.org/pdf/2310.03602v5)
- Non-photographic and technical domains with layout or spatial control:
  - charts: ChArtist (CVPR 2026, arXiv 2603.14209); Semantic-Structural Alignment for Generative Pictorial Charts (ACM TOG 2026, arXiv 2606.06498)
  - STEM instruction illustrations: LEARN, story-driven layout-to-image (arXiv 2508.11153)
  - SVG-like control: "Controlling Your Image via Simplified Vector Graphics" (arXiv 2602.14443)
  - scene graphs: SimGraph (arXiv 2601.21498)
  - medical: glioblastoma MRI with spatial/compositional control (arXiv 2611.06540); GESTALT histopathology (WACV 2027, arXiv 2611.01141, code public)
  - industrial: crack-data synthesis pipeline (arXiv 2609.12431); GroundingAnomaly (arXiv 2604.08301); long-tail detection via layout-instruction-conditioned synthesis (ICME 2026, arXiv 2611.07553)
  - degraded imagery: FICGen, layout-driven degraded image generation (ICCV 2025, arXiv 2509.01107)

  [L] — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- **Few-shot atypical layouts:** "Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation" (ICML 2026, arXiv 2605.31266). NEWEST. [L] — [README](https://raw.githubusercontent.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models/main/README.md)
- GenEval's authors caution that photo-trained detectors **do not generalise to visually distinct (art) imagery**. This is direct evidence that evaluation must also be adapted when transferring domains. [V] — [GenEval](https://arxiv.org/pdf/2310.11513)

### Inferences
- **Adopt the AeroGen/GeoDiffusion evaluation recipe.** Remote sensing is the best methodological template for archaeology. It shares top-down views, rotated and elongated objects (walls resemble roads and buildings), many small instances, and scarce labelled data. Concretely: (i) generate from held-out real layouts; (ii) run a detector trained on real domain data and report AP50 against the input boxes; (iii) optionally train a detector on synthetic data and report mAP gains on real data as a downstream-utility metric.
- **Rotated boxes.** Rotated boxes (AeroGen) are a better representation than axis-aligned ones for walls and elongated features in plan view.
- **Gap in architecture.** Floor-plan work shows the evaluation gap the project could fill: there is no standard geometric metric (wall validity, adjacency, IoU vs plan), only user studies. Segmentation-mIoU against the input plan (ControlNet++ protocol) would be a meaningful contribution.

### Gaps
- I found no layout-conditioned diffusion work evaluated on **archaeological** or cultural-heritage excavation imagery (out of scope here, but its absence is worth noting).
- WAFFLE and Qiu et al. quantitative results were not retrieved; only descriptions from search snippets and review summaries were available.
- Code and weights status for most 2026 domain-transfer entries was not checked.

### Summary table: methods, stages, representations, base models, code, metrics

Legend: T->L = Text->Layout; L->I = Layout->Image; Eval = benchmark/metric. "Yes" means public code was verified this session or is well known [K]. "?" means not verified.

| Method (venue, year, arXiv) | Stage | Layout representation | Base model | Code | Evaluation metric used |
|---|---|---|---|---|---|
| LayoutGPT (NeurIPS'23, 2305.15393) | T->L (+L->I via GLIGEN) | Boxes as CSS-style text; 3D scenes | GPT-3.5/4 + GLIGEN (SD1.4) | Yes [K] | NSR-1K counting/spatial acc.; GLIP detection |
| LMD / LMD+ (TMLR'24, 2305.13655) | T->L + L->I | Captioned boxes + background prompt | GPT-4 + SD (training-free) / GLIGEN | Yes [K] | 400-prompt benchmark, OWL-ViT detection |
| LayoutLLM-T2I (ACM MM'23, 2308.05095) | T->L + L->I | Boxes + phrases | ChatGPT + SD-based relation-aware L2I | Yes [K] | FID, CLIP sim, layout metrics [K] |
| VPGen / VPEval (NeurIPS'23, 2305.15328) | T->L + L->I + Eval | Objects, counts, boxes (step-wise) | Vicuna-13B + GLIGEN | Yes [K] | VPEval (detector/OCR/VQA/depth programs) |
| Control-GPT (2023, 2305.18583) | T->L + L->I | TikZ code sketches + grounding tokens | GPT-4 + ControlNet | ? (none found) | Spatial-relation accuracy [K] |
| SceneGenie (ICCVW'23, 2304.14573) | T->L + L->I | Scene graph -> boxes/masks | SD + guidance | ? | FID/IS on VG, COCO-Stuff [K] |
| Attention Refocusing (CVPR'24, 2306.05427) | T->L + L->I | LLM boxes | GPT-4 + SD / GLIGEN (training-free) | Yes [K] | HRS-Bench, DrawBench |
| LLM Blueprint (ICLR'24, 2310.10640) | T->L + L->I | Boxes + object descriptions + background | LLM + SD | Yes [K] | CLIP/human, object recall [K] |
| SLD (CVPR'24, 2311.16090) | T->L + L->I (closed loop) | Boxes; latent edit ops | LLM + OWL-ViT + SD/SDXL | Yes [K] | LMD benchmark (detector-based) |
| Ranni (CVPR'24, 2311.17002) | T->L + L->I | "Semantic panel" (box, colour, keypoints, text) | Fine-tuned LLM + SD | Yes [K] | Spatial/attribute/count accuracy [K] |
| RPG (ICML'24, 2401.11708) | T->L + L->I | Rectangular region partition + sub-prompts | MLLM + SDXL (training-free) | Yes [K] | T2I-CompBench |
| RealCompo (NeurIPS'24, 2402.12908) | T->L + L->I | LLM boxes / keypoints / seg | SD1.5/SDXL + GLIGEN/InstanceDiffusion | Yes [K] | T2I-CompBench |
| GenArtist (NeurIPS'24, 2407.05600) | T->L + L->I (agent) | Plan tree + boxes | MLLM + tool library | ? | T2I-CompBench, editing benchmarks [K] |
| SLayR (arXiv 2024/25, 2412.05003) | T->L | Scene layout (boxes) | Rectified-flow transformer | ? (none on abs) | Plausibility/variety metrics + human eval [V] |
| PlanGen (ICCV'25, 2503.10127) NEW | T->L + L->I unified | Layout tokens in AR VLM | Unified AR VLM | ? | ? |
| GoT (2025, 2503.10639) NEW | T->L + L->I | Reasoning chain with boxes | Qwen2.5-VL + SDXL-based [K] | Yes [K] | GenEval, editing [K] |
| Two-Stage LLM+Diffusion (arXiv Nov'25, 2511.06888) NEW | T->L + L->I | Structured layout (LLM + rule-based completion) | LLM + ControlNet vs GLIGEN, domain fine-tuned | ? (none on abs) | Object recall (57.2%->99.9%) [V] |
| ATLAS "Think, Plan, Paint" (NeurIPS'26, 2607.16409) NEWEST | T->L + L->I unified | Explicit layout shared across stages | Unified MLLM 7B/80B, RL layout alignment | ? | ATLAS-Reasoning + standard T2I benchmarks [V] |
| Tell2Design (ACL'23, 2311.15941) | T->L (floor plans) | Room polygons/boxes | Seq2Seq (T5-style) [K] | Yes [K] | IoU vs GT plan [K] |
| HouseDiffusion (CVPR'23, 2211.13287) | Graph->Layout | Vector polygons from bubble graph | Transformer diffusion | Yes [K] | FID, diversity, compatibility [K] |
| ControlNet (ICCV'23, 2302.05543) | L->I | Dense maps: seg, scribble, lineart, MLSD, depth, edges, pose | SD1.5 (community SDXL/SD3/FLUX ports) | Yes | Qualitative + user study; FID/CLIP (seg) [K] |
| T2I-Adapter (AAAI'24, 2302.08453) | L->I | Dense maps (sketch, depth, seg, pose) | SD1.4/1.5, SDXL | Yes | FID, CLIP [K] |
| ControlNet-XS (ECCV'24, 2312.06573) | L->I | Dense maps | SD1.5/2.1/SDXL | Yes [K] | FID, CLIP, condition fidelity [K] |
| ControlNet++ (ECCV'24, 2404.07987) | L->I + Eval | Dense maps | SD1.5 | Yes [K] | mIoU (seg), SSIM (lineart/HED), F1 (Canny), RMSE (depth), FID |
| ALDM (ICLR'24, 2401.08815) | L->I | Segmentation map | SD + ControlNet + segmenter feedback | Yes [K] | mIoU vs input map, FID |
| GLIGEN (CVPR'23, 2301.07093) | L->I | Boxes + phrases (also keypoints, depth, seg) | SD1.4 frozen + gated attn | Yes (diffusers) | YOLOv4 AP/AP50 on COCO; GLIP on LVIS; FID |
| LayoutDiffusion (CVPR'23, 2303.17189) | L->I | Boxes + closed-set classes | Custom LDM (COCO-Stuff/VG) | Yes [K] | FID, CAS, YOLOScore, LPIPS diversity |
| BoxDiff (ICCV'23, 2307.10816) | L->I (training-free) | Boxes | SD1.4 / +GLIGEN | Yes [K] | YOLO AP, T2I-Sim [K] |
| Layout-guidance (WACV'24, 2304.03373) | L->I (training-free) | Boxes | SD | Yes [K] | VISOR, COCO AP [K] |
| MultiDiffusion (ICML'23, 2302.08113) | L->I (training-free) | Region masks/boxes | SD | Yes [K] | Region IoU, FID [K]; baseline in COCO-MIG |
| DenseDiffusion (ICCV'23, 2308.12964) | L->I (training-free) | Segments + per-segment text | SD | Yes [K] | IoU with input masks, CLIP |
| InstanceDiffusion (CVPR'24, 2402.03290) | L->I | Points, scribbles, boxes, masks + per-instance captions | SD1.5 | Yes [K] | YOLOv8 box/mask AP on COCO, point/scribble adherence |
| MIGC / MIGC++ (CVPR'24 / TPAMI, 2402.05408 / 2407.02329) | L->I + Eval | Boxes (+masks, ref images in ++) + attributes | SD1.4 | Yes [K] | COCO-MIG: ISR, mIoU (GroundingDINO) |
| HiCo (NeurIPS'24, 2410.14324) | L->I | Boxes + captions (hierarchical) | SD1.5/SDXL [K] | Yes (list) | HiCo-7K, FID, detector AP [K] |
| Rich-context L2I (NeurIPS'24, 2409.04847) | L->I + Eval | Boxes + long descriptions | ? | ? (none on abs) | Two open-vocab metrics + user study [V] |
| ROICtrl (CVPR'25, 2411.17949) | L->I | Boxes (ROI) + captions | SD1.5/SDXL community models [K] | Yes [K] | ROICtrl-Bench, AP [K] |
| CreatiLayout / SiamLayout (ICCV'25, 2412.03859) NEW | L->I + Eval | Boxes + rich entity text | SD3/SD3.5 MM-DiT (FLUX variant) | Yes (list) | LayoutSAM-Eval (VLM-judged spatial/attr), quality |
| Regional Prompting FLUX (2024, 2411.02395) NEW | L->I (training-free) | Region masks + prompts | FLUX.1 | Yes [K] | Qualitative [K] |
| 3DIS / 3DIS-FLUX (ICLR'25 / 2501.05131) NEW | L->I (training-free renderer) | Boxes -> depth map -> FLUX depth render | SD-depth / FLUX.1-Depth-dev | Yes [V] | COCO-MIG ISR (+41% vs MultiDiffusion) |
| EliGen (2025, 2501.01097) NEW | L->I | Entity masks + prompts | FLUX.1 | Yes [K] | Entity-level metrics [K] |
| DreamRenderer (ICCV'25, 2503.12885) NEW | L->I (training-free) | Boxes/masks + per-instance attributes | FLUX.1 | ? | COCO-MIG-style ISR [K] |
| EasyControl (ICCV'25, 2503.07027) NEW | L->I | Dense maps | FLUX (DiT) LoRA | Yes [K] | Condition fidelity (ControlNet++-style) [K] |
| FLUX.1 Tools (BFL, Nov'24) NEW | L->I | Canny / depth | FLUX.1 | Weights (dev, non-commercial licence) [K] | — |
| InstanceAssemble (NeurIPS'25, 2509.16691) NEW | L->I + Eval | Boxes + text or visual content | DiT T2I via LoRA (SD3/FLUX, unverified) | Yes [V] | DenseLayout + Layout Grounding Score |
| AeroGen (CVPR'25, 2411.15497) NEW | L->I (remote sensing) | Horizontal + rotated boxes | Fine-tuned LDM | Yes [V] | Downstream detector mAP on DIOR/DIOR-R/HRSC |
| GeoDiffusion (ICLR'24, 2306.04607) | L->I (driving) | Boxes encoded as text tokens | SD fine-tuned | Yes [K] | Detector mAP (train on synthetic), FID |
| WAFFLE (arXiv 2412.00955) | L->I (floor plans) | Building layout maps (CubiCasa5K) | SD fine-tuned + ControlNet | ? | User study (36 pairs) [V] |
| LayoutBench / IterInpaint (CVPRW'24, 2304.06671) | Eval (+L->I) | Boxes, OOD CLEVR-style | — | Yes [K] | Layout AP on number/position/size/shape |
| T2I-CompBench(++) (NeurIPS'23 D&B / TPAMI'25, 2307.06350) | Eval | Text prompts | — | Yes [K] | UniDet spatial; BLIP-VQA attributes; CLIPScore |
| GenEval (NeurIPS'23 D&B, 2310.11513) | Eval | Text prompts | — | Yes [K] | Mask2Former detection: counting, position, colour |
| GenEval 2 (arXiv Dec'25, 2512.16853) NEW | Eval | Text prompts | — | ? | Soft-TIFA (VQA); shows GenEval drift up to 17.7% |
| HRS-Bench (ICCV'23, 2304.05390) | Eval | Text prompts | — | Yes [K] | UniDet counting/spatial + other skills |
| VISOR / SR2D (2022, 2212.10015) | Eval | Two-object relation prompts | — | Yes [K] | VISOR (OWL-ViT detection + relation rule) |
| DrawBench (NeurIPS'22, 2205.11487) | Eval | 200 prompts, 11 categories | — | Prompts public | Human pairwise preference |
| TIFA / DSG / VQAScore (ICCV'23 / ICLR'24 / ECCV'24) | Eval | Text prompts | — | Yes [K] | VQA-based faithfulness |
| 7Bench (ICIAP'25, 2508.12919) NEW | Eval | Boxes + prompts, 7 scenarios | — | Yes [V] | Text alignment + layout alignment score |
| C-Bench / O-Bench (CVPR'26 Findings, 2604.25358) NEWEST | Eval | Boxes + prompts (closed & open set) | — | Yes [V] | OWLv2 Acc@IoU AUC + TIFA (BLIP-2), human-validated |
| CMMD (CVPR'24, 2401.09603) | Eval | — | — | Yes [K] | CLIP-embedding MMD (FID alternative) |
