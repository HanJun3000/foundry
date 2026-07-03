# Work Summary — RF3 GPCR Fine-tuning Setup

Prepared 2026-07-02. Companion files: `plan.md` (the step-by-step plan with per-step status)
and `HPC_INSTRUCTIONS.md` (how to run the remaining HPC steps).

## Goal
Fine-tune RoseTTAFold3 (RF3) on GPCR structures, training **only the diffusion module**
(Stage A) and then **only the confidence heads** (Stage B), with the input embedders and
Pairformer/MSA trunk frozen. This is a **two-stage** fine-tune because the codebase trains
those two pieces with different objectives/mechanics (denoising loss vs. mini-rollout
confidence loss).

## Environments (important — the system python lacks the packages)
- **AtomWorks / data prep**: `/home/iwe35/projects/euro_workshop/aw_venv/bin/python`
  (atomworks 2.2.0, biotite 1.4.0, pandas). Prints a harmless `PDB_MIRROR_PATH not set` banner.
- **RF3 + foundry (model/training/config validation)**: `/home/iwe35/anaconda3/envs/rfd3_env/bin/python`
  (has both `rf3` and `foundry`; aw_venv does NOT). Used for all config/model validation below.
- **MSA tools**: `/home/iwe35/anaconda3/envs/colabfold/bin/` (mmseqs/colabfold) and
  hhblits at `/software/all/HH-suite/3.3.0-gompi-2023a/bin/hhblits` (on the HPC).

## Key discovery
The notebook `rf3_finetune_gpcr.ipynb` claims RF3 is "not installed" and its training loop is
illustrative. **That is outdated** — the full RF3 source (model, trainers, losses, Hydra configs)
is present under `rf3/src/rf3/` and `rf3/configs/`, and an installed copy lives in `rfd3_env`.
So we fine-tune through the real Hydra training stack; the notebook is only a data-prep reference.

---

## What was produced

### New files
| File | Purpose |
|---|---|
| `preprocess_gpcr.py` | Resumable data-prep pipeline (Steps 1–6a): fetch → download → annotate → manifest → split → MSA-prep |
| `hpc_generate_msas_hhblits.sh` | **Active** MSA generation script for the HPC (hhblits vs. UniRef30) |
| `hpc_generate_msas.sh` | Alternative MSA script (colabfold_search; unused — needs ColabFold mmseqs DBs) |
| `rf3/configs/datasets/gpcr.yaml` | GPCR train+val dataset config (Step 8) |
| `rf3/configs/experiment/gpcr_finetune_diffusion.yaml` | Stage A experiment (Step 10) |
| `rf3/configs/experiment/gpcr_finetune_confidence.yaml` | Stage B experiment (Step 11) |
| `plan.md`, `SUMMARY.md`, `HPC_INSTRUCTIONS.md` | Plan, this summary, HPC runbook |

### Modified source code
**`rf3/src/rf3/trainers/rf3.py`** — added a configurable parameter-freezing capability to `RF3Trainer`
(this is the repo = deployment source; the identical installed copy in `rfd3_env` was left untouched
and tested via `sys.path`). Three changes:

1. New `__init__` keyword arg (with docstring):
   ```python
   freeze_module_name_substrings: list[str] | None = None,
   ```
   stored as `self.freeze_module_name_substrings = freeze_module_name_substrings`.

2. `construct_model()` now calls the new helper after building the (optionally EMA-wrapped) model:
   ```python
   self.initialize_or_update_trainer_state({"model": model})
   # ... optionally freeze sub-modules for staged fine-tuning (e.g., diffusion-only)
   self._freeze_matching_parameters()
   ```

3. New helper method:
   ```python
   def _freeze_matching_parameters(self) -> None:
       """Freeze parameters whose name contains any of `freeze_module_name_substrings`.
       Substring matching is robust to the EMA `model.` name prefix. Frozen parameters are
       safely ignored by the optimizer (AdamW skips `grad is None`) and excluded from DDP
       gradient reduction (DDP only reduces requires_grad=True params), so no optimizer/DDP
       change is needed."""
       substrings = self.freeze_module_name_substrings
       if not substrings:
           return
       model = self.state["model"]
       n_frozen_tensors = n_frozen_params = n_trainable_params = 0
       for name, param in model.named_parameters():
           if any(s in name for s in substrings):
               param.requires_grad = False
               n_frozen_tensors += 1
               n_frozen_params += param.numel()
           elif param.requires_grad:
               n_trainable_params += param.numel()
       assert n_frozen_tensors > 0, (
           f"freeze_module_name_substrings={substrings} matched no parameters; "
           "check the substrings against the model's parameter names."
       )
       ranked_logger.info(
           f"Froze {n_frozen_tensors} parameter tensors ({n_frozen_params:,} params) "
           f"matching {substrings}; {n_trainable_params:,} params remain trainable."
       )
   ```

**No other source files were modified.** The existing `RF3TrainerWithConfidence` already freezes
everything except `confidence_head`, so Stage B needs no code change.

Why no optimizer change was needed: freezing runs in `construct_model()` which `rf3/train.py` calls
*before* `construct_optimizer()`. Frozen params passed to AdamW are skipped (`grad is None`) and DDP
only reduces `requires_grad=True` params — the same reason the existing confidence trainer works.

---

## Data produced (`data/gpcr/`)
| Path | Contents |
|---|---|
| `meta/structures.parquet` | 1716 GPCRdb structures, filtered → `keep` flag (778 kept) |
| `meta/residues.parquet` | BW generic numbers + TMH segments per receptor (81,473 rows) |
| `meta/contacts.parquet` | Per-ligand contact residues (36,551 rows) |
| `raw_cif/` | 778 downloaded mmCIFs (728 MB) |
| `annotated_cif/` | 778 cleaned + annotated CIFs (679 MB) |
| `gpcr_manifest.parquet` | 778 rows: composition, cluster, split, metadata, receptor seq |
| `gpcr_manifest_train.parquet` / `gpcr_manifest_val.parquet` | 679 / 99 rows (disjoint clusters) |
| `msa_input/queries.fasta` | 920 unique protein sequences (headers = 11-char hash) for MSA gen |

### Filtering & split
- **Kept 778 / 1716**: has a ligand of interest (small-molecule/peptide/protein), resolution
  ≤ 2.5 Å (X-ray) or ≤ 3.2 Å (cryo-EM). → 524 small-molecule, 254 protein/peptide.
- **Clustering**: on the GPCRdb `protein` id minus species suffix (mmseqs unavailable + receptor
  sequences are fusion-contaminated). 184 receptor clusters.
- **Split**: whole clusters, stratified by (receptor_class, dominant ligand class); every
  multi-cluster stratum keeps ≥1 cluster in train and holds out ≥1 in val. → train 679 / val 99
  (12.7%). Val covers 6/8 receptor classes; the 2 absent (F, O2) each have a single cluster so
  can't be held out. All 8 classes present in train. Reproducible via `--seed 42`.

### Per-atom annotations added (in `annotated_cif/`)
`is_interchain_contact` (int8), `is_contact_residue` (int8), `bw_number` (str), `tmh_segment` (str).
Not consumed by the current diffusion+confidence loss, but persisted for later interface-focused
evaluation and the deferred BW-feature work.

---

## Validation performed locally (in `rfd3_env`, real code paths)
- **Data path**: `PandasDataset` + `create_base_loader` + rf3's `build_af3_transform_pipeline`
  (is_inference=False) on the smallest train structure → produced `feats` + `ground_truth` +
  diffusion targets. Val (is_inference, no crop) likewise.
- **Dataset config**: `datasets/gpcr.yaml` instantiates through the real foundry assembly
  (`recursively_instantiate_datasets_and_samplers` for train → `ConcatDatasetWithID` (679) +
  `MixedSampler`; val via `assemble_val_loader_dict` structure → `PandasDataset` (99)).
- **Freeze mechanism**: instantiated a real RF3 net + EMA wrapper; the model has 4 param modules —
  `feature_initializer` (107 tensors), `recycler` (2847), `diffusion_module` (878), `distogram_head`
  (2). Freezing `["feature_initializer","recycler","distogram_head"]` leaves **exactly
  `diffusion_module` (878 tensors)** trainable.
- **Experiment configs**: both compose under Hydra (`experiment=gpcr_finetune_diffusion` /
  `...confidence`) resolving the correct trainer/model/dataset/MSA/checkpoint wiring.
- **Portability**: `gpcr.yaml` loads structures by `pdb_code` under `${GPCR_DATA_DIR}/annotated_cif`
  (not the manifest's absolute `path` column), so moving the data to the HPC only requires setting
  `GPCR_DATA_DIR` — no path rewriting. Verified train + val load this way.

---

## MSA convention (verified end-to-end)
RF3/AtomWorks locates an MSA by hashing the chain sequence:
- `hash = sha256(sequence.encode()).hexdigest()[:11]` (11 hex chars)
- sharded with `directory_depth=2` → `data/gpcr/msas/<h[0:2]>/<h[2:4]>/<hash>.a3m`
- config: `protein_msa_dirs=[{dir: data/gpcr/msas, extension: .a3m, directory_depth: 2}]`

Confirmed that `get_msa_path` resolves this layout, and that the hhblits script writes each a3m to
exactly that path (its filename is the sequence hash).

---

## Status by step (see `plan.md` for detail)
| Step | Status |
|---|---|
| 1 Fetch GPCRdb metadata | ✅ done |
| 2 Download mmCIFs | ✅ done (778/778) |
| 3 Parse/clean/annotate | ✅ done (778/778) |
| 4 Build manifest | ✅ done |
| 5 Cluster + split | ✅ done |
| 6 MSAs | 6a done (FASTA); **6b pending on HPC** |
| 7 Pretrained checkpoint | command ready; **download on HPC** (3.04 GB) |
| 8 GPCR dataset config | ✅ done + validated |
| 9 Trainer freeze | ✅ done + validated |
| 10 Stage A config | ✅ config done + composes; **run on HPC** |
| 11 Stage B config | ✅ config done + composes; **run on HPC** |
| 12 Evaluation | pending (after training) |

## Open items / watch-outs
- **GPU**: an 8 GB card (RTX 3070) is too small for RF3 training even with the trunk frozen — train
  on the HPC (A100/H100 class).
- **MACE-OMOL embeddings**: rf3's atom-level embedding channel looks up a cache at a Baker-internal
  path (`/net/tukwila/...`). Missing locally (fell back gracefully). Confirm it exists on the HPC or
  that training tolerates its absence (`p_dropout_atom_level_embeddings` already drops it 50% of the time).
- **Checkpoint architecture**: the public "latest" ckpt may include a confidence head; the Stage-A
  `WeightLoadingConfig` (COPY + REINIT fallback) ignores extra keys, so this is handled.
- **Deferred (not in scope)**: ligand-centered crop, interface-weighted loss, chi-angle pocket loss,
  Ballesteros–Weinstein token features, LoRA on the trunk.
