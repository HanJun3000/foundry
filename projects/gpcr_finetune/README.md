# RF3 GPCR Fine-tuning

Fine-tune RoseTTAFold3 (RF3) on GPCR structures, training **only the diffusion module**
(Stage A) then **only the confidence heads** (Stage B), with the input embedders and
Pairformer/MSA trunk frozen. This directory holds the data-prep pipeline, the HPC MSA scripts,
and the docs; the model/training changes live in the RF3 package and its configs (see below).

## What was added to foundry
| Change | Location |
|---|---|
| `freeze_module_name_substrings` on `RF3Trainer` (generic staged-freeze support) | `models/rf3/src/rf3/trainers/rf3.py` |
| GPCR train/val dataset config | `models/rf3/configs/datasets/gpcr.yaml` |
| Stage A experiment (diffusion-only) | `models/rf3/configs/experiment/gpcr_finetune_diffusion.yaml` |
| Stage B experiment (confidence-only) | `models/rf3/configs/experiment/gpcr_finetune_confidence.yaml` |

The trainer change is generic and self-contained (freeze any params whose name matches a
substring) — a candidate to upstream independently of the GPCR work.

## Install
```bash
# from the foundry repo root
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[rf3]'
foundry install base-models --checkpoint-dir <ckpt_dir>   # RF3 pretrained weights
```

## Reproduce the dataset
The large data (CIFs, MSAs) is **not** committed — only the small pinning artifacts under
`data/gpcr/` (manifests, `meta/*.parquet`, `msa_input/queries.fasta`) that define the exact
778-structure set and the train/val split. Regenerate the rest with:

```bash
cd projects/gpcr_finetune
AW=/path/to/atomworks/python           # an env with atomworks + biotite + pandas
$AW preprocess_gpcr.py fetch-meta      # GPCRdb metadata (or reuse committed meta/)
$AW preprocess_gpcr.py download        # mmCIFs -> data/gpcr/raw_cif/
$AW preprocess_gpcr.py annotate        # cleaned + annotated -> data/gpcr/annotated_cif/
$AW preprocess_gpcr.py manifest        # -> data/gpcr/gpcr_manifest.parquet
$AW preprocess_gpcr.py split           # -> gpcr_manifest_{train,val}.parquet (seed 42)
$AW preprocess_gpcr.py msa-prep        # -> data/gpcr/msa_input/queries.fasta
```
To reproduce the committed split exactly, skip `manifest`/`split` and keep the committed
manifests; only `download` + `annotate` (deterministic from the pinned PDB IDs) are needed to
recreate the structures.

MSAs are generated on an HPC (see below), then placed under `data/gpcr/msas/` in the
hash-sharded layout `<h0:2>/<h2:4>/<hash>.a3m`.

## Train (on a GPU cluster — an 8 GB card is too small)
```bash
export PROJECT_ROOT=/path/to/foundry
export GPCR_DATA_DIR=/path/to/data/gpcr           # defaults to $PROJECT_ROOT/projects/gpcr_finetune/data/gpcr
export RF3_CKPT=/path/to/rf3_pretrained.ckpt
cd $PROJECT_ROOT/models/rf3
python src/rf3/train.py experiment=gpcr_finetune_diffusion        # Stage A -> only diffusion_module trains
# then, with RF3_STAGEA_CKPT set to Stage A's output checkpoint:
python src/rf3/train.py experiment=gpcr_finetune_confidence       # Stage B -> only confidence_head trains
```

## Files here
- `preprocess_gpcr.py` — resumable data pipeline (stages: fetch-meta, download, annotate, manifest, split, msa-prep).
- `hpc_generate_msas_hhblits.sh` — SLURM job array: hhblits vs. UniRef30 → hash-sharded `.a3m` (the active MSA path).
- `hpc_generate_msas.sh` — alternative colabfold_search variant (needs ColabFold mmseqs DBs).
- `HPC_INSTRUCTIONS.md` — end-to-end HPC runbook.
- `SUMMARY.md` — detailed record of all work + the exact code change.
- `plan.md` — the step-by-step plan with per-step status.

## Notes
- Dataset loads structures by `pdb_code` under `${GPCR_DATA_DIR}/annotated_cif` (the manifest's
  absolute `path` column is legacy/ignored), so the data is relocatable via one env var.
- MSA lookup: `hash = sha256(sequence)[:11]`, `directory_depth=2`.
- Stage A freeze list `["feature_initializer","recycler","distogram_head"]` leaves exactly the
  diffusion module trainable; drop `distogram_head` to also train that auxiliary head.
