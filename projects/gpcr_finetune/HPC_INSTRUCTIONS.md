# HPC Runbook — RF3 GPCR Fine-tuning

How to finish the remaining steps on the HPC. Local work (Steps 1–5, 8, 9, and the Step 10/11
configs) is done and validated; see `SUMMARY.md`. What remains: generate MSAs (6b), get the
checkpoint (7), run Stage A (10) and Stage B (11), then evaluate (12).

Everything lives in your fork of **`RosettaCommons/foundry`** (branch `gpcr-finetune`). Below,
`$FOUNDRY` = the foundry repo root on the HPC. This project is at `$FOUNDRY/projects/gpcr_finetune`.

---

## 0. Get the fork onto the HPC

The code + small pinning data (manifests, GPCRdb metadata, MSA query FASTA) are committed, so a
clone is enough. The large data (structures, MSAs) is regenerated / copied separately.

```bash
git clone -b gpcr-finetune https://github.com/<you>/foundry.git $FOUNDRY
cd $FOUNDRY
```

The **training structures** (`annotated_cif/`, ~679 MB) are NOT in git. Either copy them from your
workstation, or regenerate them on the HPC (deterministic from the pinned PDB IDs):

```bash
# Option A — copy annotated CIFs into the in-repo data dir (default GPCR_DATA_DIR location):
rsync -av <workstation>:.../data/gpcr/annotated_cif  $FOUNDRY/projects/gpcr_finetune/data/gpcr/
# Option B — regenerate (needs an env with atomworks+biotite+pandas):
#   cd $FOUNDRY/projects/gpcr_finetune && <atomworks_py> preprocess_gpcr.py download && ... annotate
```
(`raw_cif/` is not needed at train time.)

## 1. Install foundry (with RF3)

```bash
cd $FOUNDRY
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[rf3]'
# sanity: the freeze arg we added must be present
python -c "import inspect, rf3.trainers.rf3 as t; \
  assert 'freeze_module_name_substrings' in inspect.signature(t.RF3Trainer.__init__).parameters; \
  print('trainer freeze arg present')"
```

The trainer change and the GPCR configs are already committed in the fork, so an editable install
runs them directly — nothing to copy.

---

## 2. Step 6b — Generate MSAs (hhblits)

Uses the AlphaFold2/hhsuite `UniRef30_2020_06` already on the cluster (no ColabFold DB download).

```bash
GPCR=$FOUNDRY/projects/gpcr_finetune
cd $GPCR/data/gpcr/msa_input          # where queries.fasta is (committed)
cp $GPCR/hpc_generate_msas_hhblits.sh .

# 2a. Edit the CONFIG block in the script and verify:
#   - HHBLITS path / `module load HH-suite/3.3.0-gompi-2023a`  (already prefilled)
#   - DB_UNIREF prefix: /software/databases/alphafold/UniRef30_2020_06/UniRef30_2020_06
#   - USE_BFD=0 (UniRef30 only; set 1 for deeper MSAs — much slower, raise --mem)
#   - SLURM resources (#SBATCH cpus/mem/time)
# Confirm the DB prefix is right (this file must exist):
ls /software/databases/alphafold/UniRef30_2020_06/UniRef30_2020_06_a3m.ffindex

# 2b. One-time split -> prints the sbatch array range (for 920 seqs: 0-45).
#     Recommended so you know the array range, but the run mode auto-builds it
#     (race-safe) if you skip this. Run it in the dir containing queries.fasta.
bash hpc_generate_msas_hhblits.sh split

# 2c. The script's #SBATCH --array default is 0-45%15 (correct for 920 seqs). If the
#     split printed a different count, update the --array line. Submit from THIS dir:
sbatch hpc_generate_msas_hhblits.sh

# 2d. When all array tasks finish, check coverage:
bash hpc_generate_msas_hhblits.sh verify        # -> "coverage: N present / M missing"
```

Output lands in `./msas/<h0:2>/<h2:4>/<hash>.a3m`. Move it into the data root:

```bash
mv msas $GPCR/data/gpcr/msas
```

Missing sequences (if any) fall back to single-sequence at train time — acceptable, but re-run the
array to fill them if coverage is low. Chains < 16 residues were intentionally not requested.

---

## 3. Step 7 — Get the pretrained checkpoint (~3 GB)

```bash
cd $FOUNDRY
foundry install base-models --checkpoint-dir ./checkpoints    # recommended
# or the direct URL:
# wget http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest.ckpt
```

---

## 4. Environment variables (needed by the configs)

```bash
export PROJECT_ROOT=$FOUNDRY                                   # required; configs live at $PROJECT_ROOT/models/rf3/configs
export GPCR_DATA_DIR=$FOUNDRY/projects/gpcr_finetune/data/gpcr  # structures + manifests + msas (this is also the default)
export RF3_CKPT=/path/to/rf3_pretrained.ckpt                  # from Section 3
# set USER if unset (used in the default log_dir path)
```

Quick pre-flight (composes the config; no training):
```bash
cd $FOUNDRY/models/rf3
python - <<'PY'
import os
from hydra import initialize_config_dir, compose
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
    cfg = compose(config_name="train", overrides=["experiment=gpcr_finetune_diffusion"])
print("composes OK:", cfg.name, cfg.trainer._target_, cfg.model.net._target_)
PY
```

Also confirm a couple of MSAs resolve (optional):
```bash
python - <<'PY'
from atomworks.ml.transforms.msa._msa_loading_utils import get_msa_path
import pandas as pd, os
d=os.environ["GPCR_DATA_DIR"]
seq=open(d+"/msa_input/queries.fasta").read().split("\n")[1]
print("resolved:", get_msa_path(seq, [{"dir":d+"/msas","extension":".a3m","directory_depth":2}]))
PY
```

---

## 5. Step 10 — Stage A: diffusion-only fine-tune

Trains ONLY the diffusion module (trunk + distogram head frozen).

```bash
cd $FOUNDRY/models/rf3
python src/rf3/train.py experiment=gpcr_finetune_diffusion
# multi-GPU: launch under your scheduler / torchrun per rf3/foundry conventions
```

What to check in the first log lines:
- `Froze <N> parameter tensors ... matching ['feature_initializer', 'recycler', 'distogram_head']; <M> params remain trainable.`
  The trainable params should all be `diffusion_module.*` (878 tensors).
- No NaNs; train loss decreasing.

Tunables (edit `configs/experiment/gpcr_finetune_diffusion.yaml`):
- `model.lr_scheduler.base_lr` (default 1e-4)
- `datasets.crop_size` (default 256), `datasets.diffusion_batch_size_train` (base 48 — lower if OOM)
- Drop `distogram_head` from `trainer.freeze_module_name_substrings` to also train that aux head.
- EMA is on by default (evaluate the EMA weights).

The checkpoint is written under the Hydra run's `output_dir` (see `configs/paths` / `hydra`). Note the
path of the best/last checkpoint — that is the input to Stage B.

---

## 6. Step 11 — Stage B: confidence-only fine-tune

Loads the Stage-A checkpoint, freezes everything except the confidence head, trains it via
mini-rollout. The fresh confidence head is auto-initialized (WeightLoadingConfig REINIT fallback).

```bash
export RF3_STAGEA_CKPT=/path/to/stageA/<best_or_last>.ckpt   # from Step 5's output_dir
cd $FOUNDRY/models/rf3
python src/rf3/train.py experiment=gpcr_finetune_confidence
```

Check: the frozen/trainable split should leave only `confidence_head.*` trainable; pTM/ipTM metrics
appear on validation.

---

## 7. Step 12 — Evaluate on held-out clusters

Validation runs automatically during training on `gpcr_manifest_val.parquet` (the 99 held-out-cluster
structures). Report **held-out-cluster** metrics, not global averages:
- Stage A: pocket / ligand-RMSD and interface lDDT.
- Stage B: confidence calibration (pLDDT vs. true lDDT, pAE vs. true error).
- Downstream design gates: ipTM ≥ 0.7, interface pLDDT ≥ 75 at contact residues, pAE at
  ligand–GPCR residue pairs.

For standalone scoring of a checkpoint, use `rf3/src/rf3/validate.py` with the same
`experiment=...` (it builds the val loaders from `datasets.val`). The per-atom GPCRdb annotations in
`annotated_cif/` (`is_contact_residue`, `bw_number`, `tmh_segment`) are available for interface-specific
metrics if you want to break results down by pocket residues.

---

## 8. Troubleshooting

- **`RF3_CKPT`/`GPCR_DATA_DIR` unset** → `oc.env` interpolation error. Export them (Section 4).
- **CIF not found** → confirm `annotated_cif/` is under `$GPCR_DATA_DIR` and contains `<PDBID>.cif`
  (loader builds `${GPCR_DATA_DIR}/annotated_cif/<pdb_code>.cif`).
- **Freeze matched no parameters** (assertion) → a substring in `freeze_module_name_substrings`
  doesn't match any param name; check against the model (`feature_initializer`, `recycler`,
  `diffusion_module`, `distogram_head`).
- **MACE-OMOL embedding cache warnings** (`/net/tukwila/...`) → the atom-level embedding channel.
  Harmless fallback if the cache is absent; provide it on the HPC if you want that channel populated.
- **OOM** → lower `datasets.diffusion_batch_size_train` and/or `datasets.crop_size`.
- **All chains single-sequence** → MSAs not found; re-check Section 2's `mv` and the depth-2 layout
  (`data/gpcr/msas/<h0:2>/<h2:4>/<hash>.a3m`).
- **Checkpoint shape mismatch** on load → the pretrained ckpt architecture differs from `model/rf3`;
  extra/missing keys are handled by `WeightLoadingConfig`, but a true shape mismatch means the model
  config must match the checkpoint (compare with `configs/experiment/pretrained/rf3.yaml`).
