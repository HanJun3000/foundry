# Plan — Fine-tune RF3 on GPCRs (diffusion module + confidence heads only)

Scope: freeze `feature_initializer` + `recycler` (trunk); train only `diffusion_module` (Stage A) then `confidence_head` (Stage B). Execute through the existing Hydra stack in `rf3/` — no hand-rolled training loop.

Data root: `data/gpcr/` with `raw_cif/  annotated_cif/  msas/  meta/  gpcr_manifest_{train,val}.parquet`.

Mark each step **done** when its "Done when" criterion is met.

---

## Step 1 — Fetch GPCRdb structure list + metadata
**Status:** done — 1716 structures → 778 kept (524 small-molecule ≤2.5 Å, 254 peptide/protein ≤3.2 Å). Wrote `meta/structures.parquet`, `residues.parquet` (BW/TMH, 81473 rows), `contacts.parquet` (36551 rows). Impl: `preprocess_gpcr.py fetch-meta`.
- Query GPCRdb REST (`https://gpcrdb.org/services/`; docs `https://docs.gpcrdb.org/web_services.html`).
- `GET /services/structure/` → all structures (PDB code, receptor, resolution, method, ligands).
- Filter: resolution ≤ 2.5 Å (small-molecule complexes), ≤ 3.2 Å (cryo-EM peptide/protein); keep only those with a bound ligand of interest.
- Cache per-structure metadata to `data/gpcr/meta/` keyed by `(pdb_id, chain, res_id)`: BW generic numbers, per-ligand contact residues, TMH assignment (TM1–7, ICL1–3, ECL1–3, H8).

**Done when:** `meta/structures.parquet` (filtered list) + `meta/annotations.parquet` (BW/contacts/TMH) exist.

## Step 2 — Download mmCIFs
**Status:** done — 778/778 downloaded from RCSB to `data/gpcr/raw_cif/`, zero failures. Impl: `preprocess_gpcr.py download`.
- For each retained PDB id, download to `data/gpcr/raw_cif/`: GPCRdb-refined coords where available, else RCSB (`https://files.rcsb.org/download/<PDBID>.cif`).
- Parallelize, rate-limit, retry with backoff, skip existing files. Log failures to `meta/download_failures.csv`.

**Done when:** `raw_cif/` holds one CIF per retained id (minus logged failures).

## Step 3 — Parse, clean, annotate
**Status:** todo
- Implement resumable `preprocess_gpcr.py` (promote notebook Parts 2.1–2.2), parallelized over `raw_cif/`.
- Per structure: `atomworks.io.parse(path, remove_waters=True, add_missing_atoms=True)`; take assembly `"1"`; drop NaN-coord atoms.
- Add per-atom `is_interchain_contact` via `biotite.structure.CellList` at 4.5 Å.
- Join `meta/annotations.parquet`: add `is_contact_residue` (bool), `bw_number` (float), `tmh_id` (int).
- Save to `data/gpcr/annotated_cif/` via `to_cif_file(..., extra_fields=[...])`.

**Status:** done — 778/778 annotated to `data/gpcr/annotated_cif/`, zero errors. Impl: `preprocess_gpcr.py annotate`. Notes: BW/TMH stored as fixed-width unicode, the two contact fields as int8 (bools round-trip through CIF as ambiguous "True"/"False" strings). Reload prepared CIFs with `atomworks.io.utils.io_utils.load_any(path, extra_fields=[...])` — NOT `parse()` (its `add_missing_atoms=True` erases extra fields). Verified on 2RH1: BW numbers, all TMH segments, 155 contact atoms present.

**Done when:** `annotated_cif/` populated; spot-check one file reloads with the extra fields. ✓

## Step 4 — Build manifest
**Status:** done — `data/gpcr/gpcr_manifest.parquet`, 778 rows, unique `example_id`, no nulls in required cols. Impl: `preprocess_gpcr.py manifest` (re-parses raw CIF with `add_missing_atoms=False`, ~0.4 s each). Columns: `example_id`(=pdb), `path`, `assembly_id`, `pdb_code`, `protein`, `receptor_class`, `n_prot/n_peptide/n_nuc/n_ligand`, `total_num_atoms_in_unprocessed_assembly`, `resolution`, `method`, `deposition_date`(=publication_date proxy), `ligand_class`, `has_signalling_protein`, `receptor_seq`. Composition sanity: 524 small-molecule / 254 protein-peptide; atom counts 1960–26493 (median 8569).
- One row per example → `gpcr_manifest.parquet`: `example_id`, `path` (→ `annotated_cif/`), `assembly_id`, composition `n_prot/n_peptide/n_nuc/n_ligand` (`PEPTIDE_MAX_RESIDUES=20`), plus `resolution`, `method`, `deposition_date`, `total_num_atoms_in_unprocessed_assembly`.
- NOTE for Step 5: `receptor_seq` (longest protein chain) can include crystallization fusion partners (e.g. T4 lysozyme in 2RH1, seqlen 500) — strip/split fusions before clustering, or cluster on the GPCRdb `protein` id.

**Done when:** `gpcr_manifest.parquet` has all rows with no null required columns. ✓

## Step 5 — Cluster + split
**Status:** done — `data/gpcr/gpcr_manifest_{train,val}.parquet`; `cluster`+`split` also persisted back into `gpcr_manifest.parquet`. Impl: `preprocess_gpcr.py split` (seed=42, val_fraction=0.12). mmseqs unavailable and receptor seqs are fusion-contaminated (T4 lysozyme etc.), so clustered on the curated GPCRdb `protein` id minus species suffix (`adrb1_human`/`adrb1_melga`→`adrb1`) — groups orthologs, prevents species leakage. Split stratified by each cluster's dominant ligand class; oversized clusters kept in train so no single receptor dominates val.
- Stratified by (receptor_class, dominant ligand class); every multi-cluster stratum keeps ≥1 cluster in train and holds out ≥1 in val.
- Result: 184 clusters → train 679 structures/157 clusters, val 99 structures/27 clusters (12.7%). Val ligand mix: 65 small-molecule + 34 protein/peptide. Val covers 6/8 receptor classes; the 2 absent (Class F, O2) each have only one receptor cluster so cannot be held out — all 8 classes present in train. Disjoint clusters verified.
- Re-run with different split via `preprocess_gpcr.py split --seed N --val-fraction F`.

**Done when:** train/val manifests exist with disjoint `cluster` sets. ✓

## Step 6 — Generate MSAs
**Status:** in progress — Step 6a done locally; Step 6b runs on the user's HPC (no local seq DB + no disk here; too many seqs for the public ColabFold API).
- **6a (done):** `preprocess_gpcr.py msa-prep` extracted 920 unique protein seqs (len≥16; 76 shorter peptides skipped → single-seq fallback) → `data/gpcr/msa_input/queries.fasta` (headers = 11-char hash) + `seq_index.tsv`.
- **6b (handed off):** HPC has AF2/hhsuite-format DBs (UniRef30_2020_06, bfd, uniref90, pdb70…), NOT ColabFold mmseqs DBs — so `colabfold_search` can't use them without a ~2 TB `setup_databases.sh` download. Chosen path: **hhblits** against the existing `UniRef30_2020_06` (hhblits at `/software/all/HH-suite/3.3.0-gompi-2023a/bin/hhblits`).
  - **Active script:** `hpc_generate_msas_hhblits.sh` — SLURM job array (920 seqs, CHUNK=20 → 46 tasks). Writes each a3m straight to the sharded path `data/gpcr/msas/<h0:2>/<h2:4>/<hash>.a3m` (no pack step; reruns skip finished seqs). Modes: `split` (one-time, prints array range), array `run`, `verify`. User edits CONFIG (DB prefix, `USE_BFD`, resources), runs `split`, `sbatch`, then copies `msas/` back to `data/gpcr/msas/`. Validated locally: split/verify/sharding correct.
  - Alt (unused): `hpc_generate_msas.sh` — colabfold_search variant, only if ColabFold mmseqs DBs are ever downloaded.
- MSA convention (verified end-to-end): `hash_sequence = sha256(seq)[:11]`; `directory_depth=2`; `get_msa_path` resolves `data/gpcr/msas/e5/10/e510c8881e4.a3m`. Packer derives the hash from each a3m's own query seq (robust to colabfold naming) — round-trip matches AtomWorks exactly.
- RF3 config (Step 8): `protein_msa_dirs=[{dir: data/gpcr/msas, extension: .a3m, directory_depth: 2}]`.

**Done when:** HPC job run + `data/gpcr/msas/` copied back; every protein chain in the manifests resolves to an `.a3m` (verify with `raise_if_missing_msa_for_protein_of_length_n`).

## Step 7 — Get pretrained checkpoint
**Status:** command ready; download on the training machine (HPC). URL verified reachable, **3.04 GB** (`Content-Length: 3038858700`). Not downloaded locally (disk-constrained + training will be on the HPC, not the 8 GB local GPU).
- `wget http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest.ckpt` (per `rf3/README.md`).
- Record the path and set it as `ckpt_config.path` in the Step 10 experiment config.

**Done when:** checkpoint file present on the training machine and path recorded.

## Step 8 — GPCR dataset config
**Status:** done — `rf3/configs/datasets/gpcr.yaml` created and validated end-to-end in the `rfd3_env` (which has rf3+foundry) via the real foundry assembly (`recursively_instantiate_datasets_and_samplers` for train, `assemble_val_loader_dict` structure for val). Train: `ConcatDatasetWithID` (679) + `MixedSampler`, produced a cropped featurized example. Val: `PandasDataset` (99), is_inference no-crop example with `atom_array`. Used `PandasDataset` + `create_base_loader` (StructuralDatasetWrapper is deprecated); cluster-inverse-size sampling weights; spatial-biased crop (0.7/0.3); MSAs point at `${GPCR_DATA_DIR}/msas` (depth 2). Data root via `GPCR_DATA_DIR` env (default local path).
- HPC watch-items surfaced during validation: (1) set `GPCR_DATA_DIR` on the HPC; (2) rf3's atom-level embedding channel looks up a MACE-OMOL cache at `/net/tukwila/...` (Baker-internal) — missing here, fell back gracefully; confirm those embeddings exist on the HPC or that training tolerates their absence (`p_dropout_atom_level_embeddings` already drops them 50% of the time).
- Original notes (for reference):
- Create `rf3/configs/datasets/gpcr.yaml` (+ `configs/datasets/train/gpcr/*`) modeled on `train/pdb/base.yaml` and `train/pdb/train_pn_unit.yaml`:
  - `dataset._target_: atomworks.ml.datasets.PandasDataset`, `data: <gpcr_manifest_train.parquet>`, `id_column: example_id`, loader `atomworks.ml.datasets.loaders.cif.create_base_loader`.
  - `transform._target_: ${datasets.pipeline_target}` inheriting `crop_size`, `n_msa`, `diffusion_batch_size_train`, `max_atoms_in_crop`, `run_confidence_head`.
  - `protein_msa_dirs` → `data/gpcr/msas` (via `paths.data.protein_msa_dirs`).
  - Weighted sampling by inverse cluster size / AF3 weights (`af3_weighted_sampling.yaml` pattern).
  - Val entry from `gpcr_manifest_val.parquet` (held-out clusters).
- Knobs: `crop_size` 256–384; keep `n_recycles_train` (4/10); `lr` 1e-4→1e-5 cosine + warm-up; EMA on; `clip_grad_max_norm` 10.0.

**Done when:** `python src/rf3/train.py experiment=... datasets=gpcr` instantiates the dataset and pulls one featurized example without error. ✓ (validated via foundry assembly, both train + val)

## Step 9 — Add configurable freeze to `RF3Trainer`
**Status:** done — added `freeze_module_name_substrings: list[str] | None` to `RF3Trainer.__init__` and a `_freeze_matching_parameters()` helper called at the end of `construct_model()` (substring match on param names, robust to the EMA `model.` prefix; asserts ≥1 match; logs frozen/trainable counts). Edited in `rf3/src/rf3/trainers/rf3.py` (repo = deployment source; the identical installed copy in `rfd3_env` was left untouched — tested via `sys.path`).
- Optimizer: no change needed. Frozen params are safely ignored — AdamW skips `grad is None`, and DDP only reduces `requires_grad=True` params (this is also why the existing `RF3TrainerWithConfidence` freeze works). Documented in the code.
- RF3 has 4 param modules: `feature_initializer` (107 tensors), `recycler` (2847), `diffusion_module` (878), `distogram_head` (2, auxiliary). **Strict diffusion-only freeze list = `["feature_initializer","recycler","distogram_head"]`** → leaves exactly `diffusion_module` trainable (verified on a real instantiated RF3 model in `rfd3_env`).

**Done when:** with the freeze list, a dry run logs only `diffusion_module.*` as trainable. ✓ (verified: 878 trainable tensors, all `diffusion_module`)

## Step 10 — Stage A: diffusion-only fine-tune
**Status:** config done, composes under Hydra; training run pending on HPC. `rf3/configs/experiment/gpcr_finetune_diffusion.yaml` created and verified to compose (`experiment=gpcr_finetune_diffusion` → `RF3Trainer` + `RF3` + freeze list + `run_confidence_head=false` + gpcr dataset/MSA/ckpt wiring). Checkpoint mismatch is handled by `WeightLoadingConfig` (COPY + REINIT fallback iterates the model's params → ignores extra ckpt keys, reinits missing ones); no non-strict-load code needed. Set envs on the HPC: `PROJECT_ROOT`, `GPCR_DATA_DIR`, `RF3_CKPT`.
- Create `rf3/configs/experiment/gpcr_finetune_diffusion.yaml` (model on `experiment/pretrained/rf3.yaml`):
  - `defaults`: `override /datasets: gpcr`, `override /model: rf3`, `override /trainer: rf3`.
  - `ckpt_config`: `foundry.utils.weights.CheckpointConfig`, `path: <pretrained ckpt>`, `reset_optimizer: true`.
  - `trainer.freeze_module_name_substrings: ["feature_initializer", "recycler", "distogram_head"]` (strict diffusion-only; drop `distogram_head` if you want its auxiliary distogram loss to keep training that 2-tensor head).
  - `datasets.run_confidence_head: false`.
- Run: `cd rf3 && python src/rf3/train.py experiment=gpcr_finetune_diffusion`.
- Verify only `diffusion_module.*` trains; loss decreases on train + held-out clusters; no NaNs.

**Done when:** Stage-A checkpoint saved and val loss improves vs. pretrained baseline on held-out clusters.

## Step 11 — Stage B: confidence-only fine-tune
**Status:** config done, composes under Hydra; training run pending on HPC (needs Stage-A checkpoint). `rf3/configs/experiment/gpcr_finetune_confidence.yaml` created and verified to compose (`experiment=gpcr_finetune_confidence` → `RF3TrainerWithConfidence` + `RF3WithConfidence` + `run_confidence_head=true`, no freeze list — the trainer freezes all but `confidence_head` itself). `WeightLoadingConfig` REINIT fallback fresh-inits the confidence head while copying the frozen trunk + Stage-A diffusion weights. Set envs on the HPC: `PROJECT_ROOT`, `GPCR_DATA_DIR`, `RF3_STAGEA_CKPT`.
- Create `rf3/configs/experiment/gpcr_finetune_confidence.yaml` (model on `experiment/pretrained/rf3_with_confidence.yaml`):
  - `defaults`: `override /model: rf3_with_confidence`, `override /trainer: rf3_with_confidence`, `override /datasets: gpcr`.
  - `datasets.run_confidence_head: true`.
  - `ckpt_config.path: <Stage-A checkpoint>`, `reset_optimizer: true`; confirm non-strict load tolerates the newly-added `confidence_head` keys (or fresh-init the head).
- Uses existing `RF3TrainerWithConfidence` (already freezes all but `confidence_head`) — no trainer code change.
- Run: `cd rf3 && python src/rf3/train.py experiment=gpcr_finetune_confidence`.
- Verify only `confidence_head.*` trains; pTM/ipTM computed on val.

**Done when:** Stage-B checkpoint saved and confidence is calibrated (pLDDT vs. true lDDT, pAE vs. true error) on held-out clusters.

## Step 12 — Evaluate on held-out clusters
**Status:** todo
- Report (never global-only): pocket/ligand-RMSD + interface lDDT (Stage A); pLDDT and pAE calibration (Stage B).
- Check downstream design gates: ipTM ≥ 0.7, interface pLDDT ≥ 75 at contact residues, pAE at ligand–GPCR residue pairs.

**Done when:** evaluation report produced for the held-out cluster split.

---

## Deferred (not in scope now — do NOT implement unless Steps 10–12 prove insufficient)
Ligand-centered spatial crop; interface-weighted coordinate loss; chi-angle pocket loss; Ballesteros–Weinstein token features (449→457-dim); LoRA on the trunk.
