# Milestone 3: Beyond the Grid (IE7615, Group 7)

**A trustworthy, explainable detector for 73 everyday objects.**
Group members: Tanmayi Shurpali, Santhoshkumar Chandrasekar, Praniti Sunil Kale, Mukul Sridar Haladhi

Milestone 2 ended with a YOLO26-s detector (`yolo26s_ft2`) that worked well on upright grids of course photos but
lost many objects when the grid was rotated, the photos were scattered, or the objects were placed on a new
background. Milestone 3 fixes that in five steps:

1. **Every box checked by a person.** All 6,881 usable photos reviewed by hand (423 boxes fixed, 14 photos excluded,
   41 near-duplicates across splits removed, 16 second-object boxes added).
2. **Harder training data.** Rotated grids, photos scattered at any angle on real room backgrounds, objects cut out
   with SAM 2 and placed on unseen backgrounds, plus dark, blurred and compressed images.
3. **Two improved models.** `yolo26s_boost` (1,280 px, batch 4) and `yolo26s_boost2` (1,600 px, batch 2), both
   fine-tuned from the Milestone 2 model for 30 epochs.
4. **Selection by a rule fixed in advance.** The rule ("higher validation mAP@0.5:0.95 wins") was written and
   SHA-256 hashed before any boost2 result existed. **boost won** (0.886 vs 0.878; paired bootstrap 95% interval of
   the difference [-0.0142, -0.0024]). Grid and scene thresholds were also chosen on validation.
5. **One look at five test sets**, then The EYE v3 to explain every result.

## Final results

Selected model **yolo26s_boost**, 1,280 px; grids: threshold 0.10, one box per 224-px cell; scenes: threshold 0.35
with class-agnostic duplicate removal at IoU 0.5. Same corrected truth for both models (report Table 9).

| Test set | Images / objects | Model | Recall | Precision | Wrong IDs | Fully correct | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|---|
| Normal grids | 72 / 1,048 | M2 model | 0.912 | 0.950 | 16 | 0.347 | 0.947 | 0.873 |
| | | **Final** | **0.949** | **0.970** | **10** | **0.472** | **0.971** | **0.900** |
| Rotated grids | 216 / 3,144 | M2 model | 0.756 | 0.845 | 230 | 0.093 | 0.813 | 0.714 |
| | | **Final** | **0.922** | **0.947** | **58** | **0.347** | **0.953** | **0.863** |
| Scattered scenes | 104 / 1,039 | M2 model | 0.767 | 0.830 | 33 | 0.086 | 0.803 | 0.635 |
| | | **Final** | **0.897** | **0.943** | **17** | **0.250** | **0.936** | **0.803** |
| Cut-outs | 104 / 1,014 | M2 model | 0.452 | 0.799 | 70 | 0.000 | 0.559 | 0.527 |
| | | **Final** | **0.881** | **0.914** | **61** | **0.288** | **0.912** | **0.887** |
| Single photos | 1,040 / 1,048 | M2 model | 0.905 | 0.954 | 12 | 0.904 | 0.941 | 0.865 |
| | | **Final** | **0.951** | **0.968** | **11** | **0.951** | **0.970** | **0.894** |

Sources: `outputs/selection/selection.md` (final settings) and `outputs/boost2/evaluation.json` (M2 model as
"before", boost as "previous"; mAP does not depend on the threshold).

**Note on comparing with Milestone 2:** the test truth was corrected in this milestone (1,048 test objects instead of
1,041), so the "M2 model" row here differs slightly from the Milestone 2 report (e.g. recall 0.912 vs 0.918). Both are
correct for their truth; every comparison in this folder uses the corrected truth.

## Repository structure

```
Milestone_3/
├── README.md
├── requirements.txt
├── scripts/                              # the extension pipeline, in the order it was run
│   ├── m2_diagnose.py                    # 1. validation diagnosis: why objects are missed, rotation baseline
│   ├── m2_label_check.py                 # 2. review tool: prepare / review / apply / merge (pass 1 and pass 2)
│   ├── m2_boost.py                       # 3. SAM 2 cut-outs, training data, test sets, training, evaluation
│   └── m2_boost_val.py                   # 4. model selection by the fixed rule, grid/scene thresholds, one look at tests
├── app/                                  # The EYE (local web app)
│   ├── m2_eye_app_v3.py                  # The EYE v3 (port 7862), the final version
│   ├── m2_eye_app_v2.py                  # v2 (port 7861), kept unchanged as a fallback
│   ├── launch_eye.py                     # fixes the weight paths for the machine it runs on, then starts v3
│   ├── RUN_EYE_README.txt                # setup and folder layout The EYE expects
│   └── requirements.txt
├── live_demo/                            # the public upload-only demo on Hugging Face (see below)
│   ├── app.py, object_names.json, requirements.txt, packages.txt, README.md
├── docs/
│   ├── The_EYE_user_manual.html          # every control, colour and panel of The EYE v3
│   └── MILESTONE3_NARRATIVE.txt          # step-by-step account of the milestone
├── outputs/
│   ├── label_check/                      # the human check of every box
│   │   ├── boxes_master_checked.csv      #   final checked box file (after pass 2)
│   │   ├── decisions.csv                 #   every review decision, time-stamped (replayable)
│   │   ├── *_pass1_snapshot.csv          #   the same two files after pass 1
│   │   ├── extra_boxes.csv               #   16 second-object boxes
│   │   ├── near_duplicates.json          #   candidate near-duplicate pairs across splits
│   │   ├── model_opinion*.json           #   the model's second opinion used to rank photos (pass 1, pass 2)
│   │   ├── summary.json                  #   6,881 checked, 0 unchecked, 14 excluded, 41 duplicates removed
│   │   └── master_report_before_check.json  # label coverage before the check (for comparison)
│   ├── boost/, boost2/                   # train.json and evaluation.json of each improved model
│   ├── selection/                        # selection.json / .md, selected_model.json, cached predictions
│   ├── test_sets_truth/                  # truth.json of every test set and the validation set (boxes only)
│   └── integrity/protected_hashes_before.txt  # SHA-256 of the protected Milestone 2 files
└── report/
    └── IE7615_Milestone3_Report_Group7.pdf
```

## Reproducibility and safeguards

* **Protected Milestone 2 files.** The Milestone 2 model, `results.json`, `boxes_master.csv` and `m2_eye_app.py`
  were hashed before the extension and re-checked after every phase. The hashes in
  `outputs/integrity/protected_hashes_before.txt` match the copies in this repository
  (`../Milestone_2/final_pipeline/outputs/results/results.json`, `../Milestone_2/final_pipeline/app/m2_eye_app.py`)
  and the report's Table 13.
* **Fixed selection rule.** `m2_boost_val.py` refuses to run if the rule file no longer matches its recorded hash
  (rule SHA-256 `C6A67BD9…F8C0`, script SHA-256 `6250EFDE…5442`, both in the report's Table 13).
* **Resumable, logged steps.** Training resumes from its last checkpoint; every review click is in `decisions.csv`.
* **One look at each test set,** after all choices were fixed on validation.

## How to run

All scripts find the project folder by themselves (the nearest folder above them that contains
`milestone2_outputs/`, or `M2_PROJECT_DIR`) and write only to `M2 extra/`. They need the course dataset and the
Milestone 2 outputs, which are not in this repository. Hardware used: NVIDIA RTX 5070 Laptop GPU (8 GB), PyTorch 2.11,
Ultralytics 8.4.

```
pip install -r requirements.txt
python scripts/m2_diagnose.py                       # validation diagnosis
python scripts/m2_label_check.py prepare            # model's second opinion on every photo
python scripts/m2_label_check.py review             # review page in the browser (stop and resume any time)
python scripts/m2_label_check.py apply              # writes the checked box files
python scripts/m2_boost.py masks                    # SAM 2 cut-outs (optional)
M2_IMGSZ=1280 M2_BATCH=4 python scripts/m2_boost.py all                       # build -> train -> evaluate (boost)
M2_BOOST_DIR=boost2 M2_IMGSZ=1600 M2_BATCH=2 python scripts/m2_boost.py all   # the second model (boost2)
python scripts/m2_boost_val.py                      # selection by the fixed rule, thresholds, one look at tests
python app/launch_eye.py                            # The EYE v3 at http://127.0.0.1:7862
```

## Live demo

**The EYE v3 (full app): https://leaves-ladies-reminder-becoming.trycloudflare.com**

The complete EYE v3 runs on our laptop with its GPU and is shared through a Cloudflare quick tunnel (report
Section 10.1). It has every feature: uploads, random grids, rotated grids, scattered and cut-out scenes, right/wrong
scoring, the stress test and the inspect panels. The link works while the app and the tunnel are running; a quick
tunnel's address changes each time it is restarted, so the current address is kept here.

**Towards permanent hosting on Hugging Face.** In future we plan to host the entire EYE on Hugging Face. As a first
step we deployed an upload-only version there:
**https://huggingface.co/spaces/TanmayiShurpali/the-eye**

It runs on Hugging Face's free ZeroGPU with the selected model and the same validated rules as The EYE v3 (grids:
threshold 0.10, one box per 224-px cell; scenes: threshold 0.35 with duplicate removal). Anyone can upload an image
and see every detected Object ID with its box and confidence. What this exploration showed, and what full hosting
still needs:
* **Hosting type.** Free GPU hosting on Hugging Face supports Gradio apps only; The EYE v3 is its own web server, so
  the full app needs a Docker Space, which requires a paid plan and runs on CPU (slower detections).
* **Privacy.** The EYE's grids, scenes, stress test and scoring use the course photos. Publishing them permanently
  needs the permission of every student who contributed photos (report Section 10.2).
* **Ready to move.** A self-contained bundle for a Docker Space was prepared (`make_space_bundle.py`, report
  Section 10.2), so the full app can follow once both conditions are met.

The upload-only app's code is in `live_demo/`; its model weights are uploaded to the Space only.

## Not included in this repository

* **The dataset and all generated images** (photos, grids, scenes, cut-outs, SAM 2 masks, preview and review sheets).
  They are student-collected course data; the TAs asked us to push code only. `OBJ054_017` (a phone screenshot with
  personal contact details) stays excluded everywhere.
* **Model weights (`.pt`)** for all models. The selected model's weights are only on the Hugging Face Space.
* **The object list with student names.** `live_demo/object_names.json` keeps only Object IDs and object names.
* **Sharing and hosting scripts** listed in the report's Appendix A (`start_share.ps1`, `stop_share.ps1`,
  `make_space_bundle.py`, `upload_space.py`) and the selection rule file (`selection_rule.txt` with its hash file);
  they remain on the machine where the work was run.
