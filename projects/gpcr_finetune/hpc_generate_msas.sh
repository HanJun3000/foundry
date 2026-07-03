#!/bin/bash
# =============================================================================
# Step 6b (runs on the HPC): generate per-sequence MSAs for RF3 GPCR fine-tuning.
#
# Input : queries.fasta  (920 unique protein sequences; produced locally by
#         `preprocess_gpcr.py msa-prep`, headers are the 11-char sequence hash)
# Output: msas/<h0:2>/<h2:4>/<hash>.a3m   (the sharded layout AtomWorks/RF3 expects,
#         hash = sha256(sequence).hexdigest()[:11], directory_depth=2)
#
# It runs ColabFold's mmseqs2 search against a LOCAL sequence DB, then packs the
# results into the hash-sharded layout. No AtomWorks needed on the HPC.
#
# HOW TO USE
#   1) Copy this repo's  data/gpcr/msa_input/queries.fasta  to the HPC.
#   2) Edit the CONFIG block below (conda/module load, DBBASE, DB names).
#   3) Submit:   sbatch hpc_generate_msas.sh       (or: bash hpc_generate_msas.sh)
#   4) Copy the resulting  msas/  directory back to  data/gpcr/msas/  in this repo.
# =============================================================================

#SBATCH --job-name=gpcr_msa
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=gpcr_msa_%j.log
# (Adjust the #SBATCH directives above to your cluster. Add --gres=gpu:1 if using --gpu 1.)

set -euo pipefail

# ============================== CONFIG (EDIT ME) =============================
# --- Activate the ColabFold environment (choose ONE; edit paths) ------------
source /home/iwe35/anaconda3/etc/profile.d/conda.sh && conda activate colabfold
# module load colabfold           # ... or your cluster's module system

# --- Local sequence database (the whole reason this runs on the HPC) --------
# DBBASE is the directory created by ColabFold's setup_databases.sh; it holds
# the mmseqs DBs. Set DB names to match what you have. Set USE_ENV=0 if you only
# have UniRef30 (no colabfold_envdb).
DBBASE="/path/to/colabfold_db"                     # <-- EDIT: directory containing the DBs
DB_UNIREF="uniref30_2302_db"                       # <-- EDIT to your UniRef30 db name
DB_ENV="colabfold_envdb_202108_db"                 # <-- EDIT (only used if USE_ENV=1)
USE_ENV=1                                          # 1 if you have the env DB, else 0

# --- Compute ----------------------------------------------------------------
THREADS="${SLURM_CPUS_PER_TASK:-32}"
GPU=0                                              # 1 to use GPU-accelerated mmseqs (needs --gres=gpu)
DB_LOAD_MODE=2                                     # 2 keeps DB on disk (low RAM); 0/3 preload (needs big RAM)

# --- Paths ------------------------------------------------------------------
QUERIES="${QUERIES:-queries.fasta}"               # the FASTA you copied over
WORKDIR="${WORKDIR:-cf_search_out}"               # scratch for colabfold_search
MSA_OUT="${MSA_OUT:-msas}"                         # final sharded a3m tree (copy this back)
MMSEQS="${MMSEQS:-$(command -v mmseqs)}"
# ============================================================================

echo "[cfg] queries=$QUERIES  DBBASE=$DBBASE  use_env=$USE_ENV  threads=$THREADS  gpu=$GPU"
[ -f "$QUERIES" ] || { echo "ERROR: $QUERIES not found"; exit 1; }
[ -d "$DBBASE" ]  || { echo "ERROR: DBBASE '$DBBASE' not found (edit CONFIG)"; exit 1; }
mkdir -p "$WORKDIR" "$MSA_OUT"

# ------------------------------ 1) MSA search -------------------------------
# colabfold_search runs one mmseqs pass over ALL queries at once (efficient),
# emitting one a3m per query into $WORKDIR.
echo "[1/2] colabfold_search ($(grep -c '^>' "$QUERIES") queries) ..."
colabfold_search \
  --threads "$THREADS" \
  --use-env "$USE_ENV" \
  --db-load-mode "$DB_LOAD_MODE" \
  --gpu "$GPU" \
  --mmseqs "$MMSEQS" \
  --db1 "$DB_UNIREF" \
  ${USE_ENV:+--db3 "$DB_ENV"} \
  "$QUERIES" "$DBBASE" "$WORKDIR"

# --------------------------- 2) Pack into shards ----------------------------
# Derive each file's hash from the QUERY sequence inside the a3m (the first
# record), so packing is robust to however colabfold_search names its outputs.
echo "[2/2] packing a3m files into $MSA_OUT/<h0:2>/<h2:4>/<hash>.a3m ..."
python3 - "$WORKDIR" "$MSA_OUT" "$QUERIES" <<'PYEOF'
import sys, os, gzip, hashlib, glob

workdir, msa_out, queries = sys.argv[1], sys.argv[2], sys.argv[3]

def h11(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:11]

def open_any(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)

def query_seq_from_a3m(path: str) -> str | None:
    """Return the query sequence = first record, uppercased with gaps removed."""
    with open_any(path) as fh:
        seq, in_first = [], False
        for line in fh:
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if in_first:       # reached the 2nd record -> query is complete
                    break
                in_first = True
                continue
            if in_first:
                seq.append(line.strip())
    s = "".join(seq).upper().replace("-", "").replace(".", "")
    return s or None

# expected hashes (from the FASTA headers)
expected = set()
with open(queries) as fh:
    for line in fh:
        if line.startswith(">"):
            expected.add(line[1:].strip())

written, by_hash = 0, set()
for path in sorted(glob.glob(os.path.join(workdir, "*.a3m")) +
                   glob.glob(os.path.join(workdir, "*.a3m.gz"))):
    q = query_seq_from_a3m(path)
    if not q:
        continue
    h = h11(q)
    dest_dir = os.path.join(msa_out, h[0:2], h[2:4])
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{h}.a3m")
    if not os.path.exists(dest):
        # write plain-text a3m (decompress if the source was gzipped)
        with open_any(path) as src, open(dest, "w") as out:
            out.write(src.read())
        written += 1
    by_hash.add(h)

missing = expected - by_hash
print(f"    packed {written} new a3m ({len(by_hash)} unique hashes) into {msa_out}")
print(f"    coverage: {len(by_hash & expected)}/{len(expected)} query sequences have an MSA")
if missing:
    with open(os.path.join(msa_out, "MISSING_MSAS.txt"), "w") as f:
        f.write("\n".join(sorted(missing)))
    print(f"    WARNING: {len(missing)} queries produced no MSA -> {msa_out}/MISSING_MSAS.txt")
    print("    (these chains will fall back to single-sequence at train time)")
PYEOF

echo "[done] Copy '$MSA_OUT' back to the repo at data/gpcr/msas/"
echo "       RF3 config: protein_msa_dirs=[{dir: data/gpcr/msas, extension: .a3m, directory_depth: 2}]"
