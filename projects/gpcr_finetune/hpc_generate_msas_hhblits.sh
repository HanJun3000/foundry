#!/bin/bash
# =============================================================================
# Step 6b (HPC, hhblits variant): generate per-sequence MSAs for RF3 GPCR
# fine-tuning using hhblits against an EXISTING AlphaFold2/hhsuite UniRef30 DB
# (no ColabFold mmseqs DB download needed).
#
# Input : queries.fasta  (920 unique protein seqs; headers are the 11-char hash;
#         produced locally by `preprocess_gpcr.py msa-prep`)
# Output: msas/<h0:2>/<h2:4>/<hash>.a3m   (sharded layout AtomWorks/RF3 expects,
#         hash = sha256(sequence).hexdigest()[:11], directory_depth=2)
#
# hhblits writes a3m natively and we know each sequence's hash (= its FASTA
# header), so each MSA is written straight to its final sharded path — no pack
# step, and reruns skip finished sequences.
#
# USAGE
#   0) Copy data/gpcr/msa_input/queries.fasta to the HPC (next to this script).
#   1) Edit the CONFIG block (module/HHBLITS path, DB paths, USE_BFD, resources).
#   2) One-time split:   bash hpc_generate_msas_hhblits.sh split
#      -> prints the number of chunks; set that as the sbatch --array range.
#   3) Submit array:     sbatch hpc_generate_msas_hhblits.sh
#   4) Check coverage:   bash hpc_generate_msas_hhblits.sh verify
#   5) Copy  msas/  back to  data/gpcr/msas/  in the repo.
# =============================================================================

#SBATCH --job-name=gpcr_hhblits
#SBATCH --array=0-45%15          # 920 seqs / CHUNK(20) = 46 tasks (0..45); %15 = max concurrent
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                # UniRef30 only; raise to ~64G if USE_BFD=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/hhblits_%A_%a.log

set -uo pipefail

# ============================== CONFIG (EDIT ME) =============================
# --- hhblits binary (prefer the module so HHLIB/paths are set correctly) -----
module load HH-suite/3.3.0-gompi-2023a 2>/dev/null || true
HHBLITS="${HHBLITS:-/software/all/HH-suite/3.3.0-gompi-2023a/bin/hhblits}"

# --- Databases (hhblits -d wants the DB *prefix*, no _a3m.ffdata suffix) -----
AFDB="/software/databases/alphafold"
DB_UNIREF="$AFDB/UniRef30_2020_06/UniRef30_2020_06"   # <-- verify this prefix exists (*_a3m.ffindex)
USE_BFD=0                                             # 1 = also search BFD (deeper, much slower/heavier)
DB_BFD="$AFDB/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"  # <-- verify prefix if USE_BFD=1

# --- hhblits search params (AF2-like) ---------------------------------------
N_ITER=3
EVALUE=0.001
CPU="${SLURM_CPUS_PER_TASK:-8}"

# --- Batch/paths ------------------------------------------------------------
CHUNK=20                                              # sequences per array task (keep in sync with --array)
QUERIES="${QUERIES:-queries.fasta}"
SPLIT_DIR="${SPLIT_DIR:-queries_split}"               # per-sequence fasta files
LIST="${SPLIT_DIR}/query_list.txt"                    # hash order
MSA_OUT="${MSA_OUT:-msas}"                            # final sharded a3m tree (copy this back)
# ============================================================================

mode="${1:-run}"

sharded_dest() {                                      # $1 = 11-char hash -> sharded a3m path
    local h="$1"; echo "${MSA_OUT}/${h:0:2}/${h:2:2}/${h}.a3m"
}

do_split() {                                          # build per-sequence fastas + query_list.txt
    [ -f "$QUERIES" ] || { echo "ERROR: $QUERIES not found (run from the dir containing it)"; exit 1; }
    mkdir -p "$SPLIT_DIR"
    python3 - "$QUERIES" "$SPLIT_DIR" "$LIST" <<'PYEOF'
import sys, os
queries, split_dir, list_path = sys.argv[1], sys.argv[2], sys.argv[3]
recs, h, seq = [], None, []
def flush():
    if h:
        with open(os.path.join(split_dir, f"{h}.fa"), "w") as f:
            f.write(f">{h}\n{''.join(seq)}\n")
        recs.append(h)
with open(queries) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            flush(); h = line[1:].strip(); seq = []
        elif line:
            seq.append(line)
flush()
# atomic write of the list so concurrent readers never see a partial file
tmp = list_path + ".tmp"
with open(tmp, "w") as f:
    f.write("\n".join(recs) + "\n")
os.replace(tmp, list_path)
print(f"split {len(recs)} sequences into {split_dir}/")
PYEOF
}

ensure_split() {                                      # build the split once, race-safe under a job array
    [ -f "$LIST" ] && return 0
    ( flock 9 2>/dev/null || true                     # serialize; if flock is unavailable, proceed anyway
      [ -f "$LIST" ] || { echo "[run] query_list.txt missing; building split now..."; do_split; }
    ) 9>"${SPLIT_DIR}.lock"
}

# ------------------------------- split mode ---------------------------------
if [[ "$mode" == "split" ]]; then
    do_split
    n=$(wc -l < "$LIST")
    chunks=$(( (n + CHUNK - 1) / CHUNK ))
    echo "-> $n sequences, CHUNK=$CHUNK => $chunks array tasks"
    echo "-> set:  #SBATCH --array=0-$((chunks-1))%15   (then: sbatch $0)"
    exit 0
fi

# ------------------------------ verify mode ---------------------------------
if [[ "$mode" == "verify" ]]; then
    [ -f "$LIST" ] || { echo "run 'split' first"; exit 1; }
    have=0; miss=0; : > MISSING_MSAS.txt
    while read -r h; do
        [ -z "$h" ] && continue
        if [ -s "$(sharded_dest "$h")" ]; then have=$((have+1)); else miss=$((miss+1)); echo "$h" >> MISSING_MSAS.txt; fi
    done < "$LIST"
    echo "coverage: $have present / $miss missing (see MISSING_MSAS.txt if >0)"
    exit 0
fi

# -------------------------------- run mode ----------------------------------
[ -x "$HHBLITS" ] || { echo "ERROR: hhblits not found at $HHBLITS"; exit 1; }
[ -f "${DB_UNIREF}_a3m.ffindex" ] || echo "WARN: ${DB_UNIREF}_a3m.ffindex not found - check DB_UNIREF prefix"
ensure_split                                          # auto-build the split if 'split' was not run first
[ -f "$LIST" ] || { echo "ERROR: $LIST still missing after split attempt (is $QUERIES present?)"; exit 1; }
mkdir -p "$MSA_OUT" logs

mapfile -t ALL < "$LIST"
N=${#ALL[@]}
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    start=$(( SLURM_ARRAY_TASK_ID * CHUNK )); end=$(( start + CHUNK ))
else
    echo "(no SLURM array -> processing ALL $N sequences serially; this is slow)"
    start=0; end=$N
fi
(( end > N )) && end=$N

db_args=(-d "$DB_UNIREF")
[[ "$USE_BFD" == "1" ]] && db_args+=(-d "$DB_BFD")

echo "[run] task=${SLURM_ARRAY_TASK_ID:-serial} seqs[$start:$end) of $N  cpu=$CPU  bfd=$USE_BFD"
done_n=0; skip_n=0; fail_n=0
for (( i=start; i<end; i++ )); do
    h="${ALL[$i]}"
    dest="$(sharded_dest "$h")"
    if [ -s "$dest" ]; then skip_n=$((skip_n+1)); continue; fi
    mkdir -p "$(dirname "$dest")"
    if "$HHBLITS" -i "$SPLIT_DIR/${h}.fa" -oa3m "$dest" -o /dev/null \
            -n "$N_ITER" -e "$EVALUE" -cpu "$CPU" -maxseq 1000000 "${db_args[@]}" \
            > "logs/hhblits_${h}.log" 2>&1; then
        done_n=$((done_n+1))
    else
        fail_n=$((fail_n+1)); rm -f "$dest"; echo "$h" >> "logs/failed_${SLURM_ARRAY_TASK_ID:-serial}.txt"
    fi
done
echo "[run] done=$done_n skipped=$skip_n failed=$fail_n"
echo "      when all array tasks finish: 'bash $0 verify', then copy '$MSA_OUT' -> repo data/gpcr/msas/"
