#!/usr/bin/env python
"""GPCR data preprocessing for RF3 fine-tuning (plan.md Steps 1-3).

Stages (resumable; each skips work already on disk):
  fetch-meta : Step 1 - GPCRdb structure list + per-structure BW/TMH/contact metadata
  download   : Step 2 - download filtered mmCIFs (GPCRdb-refined -> RCSB fallback)
  annotate   : Step 3 - parse/clean + interface & GPCRdb annotation -> annotated_cif/
  manifest   : Step 4 - composition + metadata table -> gpcr_manifest.parquet
  split      : Step 5 - cluster by receptor + train/val split -> gpcr_manifest_{train,val}.parquet
  msa-prep   : Step 6a - unique protein seqs -> data/gpcr/msa_input/queries.fasta (for HPC MSA gen)

Run: aw_venv/bin/python preprocess_gpcr.py <stage> [--limit N] [--workers K]
             preprocess_gpcr.py split [--val-fraction 0.12] [--seed 42]
             preprocess_gpcr.py msa-prep [--min-len 16]
Step 6b (MSA search itself) runs on an HPC via hpc_generate_msas.sh (needs a seq DB).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GPCRDB = "https://gpcrdb.org/services"
RCSB_CIF = "https://files.rcsb.org/download/{pdb}.cif"
GPCRDB_REFINED = "https://gpcrdb.org/structure/homology_models/{pdb}_refined_full/download_pdb"  # optional

DATA = Path(__file__).parent / "data" / "gpcr"
RAW = DATA / "raw_cif"
ANNOT = DATA / "annotated_cif"
META = DATA / "meta"

# Ligand classes we consider "of interest"
LIGANDS_OF_INTEREST = {"small-molecule", "peptide", "protein"}
CRYOEM_METHODS = {"Electron microscopy", "Electron crystallography"}
RES_MAX_XRAY = 2.5
RES_MAX_CRYOEM = 3.2
CONTACT_CUTOFF = 4.5  # Angstrom, interchain contact annotation


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "rf3-gpcr-finetune/1.0"})
    return s


def _get_json(sess: requests.Session, url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# Step 1: fetch metadata
# --------------------------------------------------------------------------- #
def _classify_and_filter(structs: list[dict]) -> pd.DataFrame:
    rows = []
    for s in structs:
        ligs = s.get("ligands") or []
        types = {l.get("type") for l in ligs} - {"none"}
        interest = types & LIGANDS_OF_INTEREST
        method = s.get("type")
        res = s.get("resolution")
        is_cryoem = method in CRYOEM_METHODS
        thr = RES_MAX_CRYOEM if is_cryoem else RES_MAX_XRAY
        keep = bool(interest) and res is not None and res <= thr
        has_pep_prot = bool(types & {"peptide", "protein"})
        ligand_class = "protein/peptide" if has_pep_prot else ("small-molecule" if "small-molecule" in types else "other")
        rows.append({
            "pdb_code": s["pdb_code"].upper(),
            "protein": s.get("protein"),
            "receptor_class": s.get("class"),
            "family": s.get("family"),
            "species": s.get("species"),
            "preferred_chain": s.get("preferred_chain"),
            "resolution": res,
            "method": method,
            "state": s.get("state"),
            "publication_date": s.get("publication_date"),
            "ligand_types": ",".join(sorted(types)) if types else "",
            "ligand_class": ligand_class,
            "has_signalling_protein": bool(s.get("signalling_protein")),
            "res_threshold": thr,
            "keep": keep,
        })
    return pd.DataFrame(rows)


def stage_fetch_meta(limit: int | None, workers: int) -> None:
    META.mkdir(parents=True, exist_ok=True)
    sess = _session()

    print("[1] fetching structure list ...")
    structs = _get_json(sess, f"{GPCRDB}/structure/")
    assert structs, "failed to fetch structure list"
    df = _classify_and_filter(structs)
    df.to_parquet(META / "structures.parquet")
    kept = df[df.keep].reset_index(drop=True)
    print(f"    total={len(df)}  kept={len(kept)}  "
          f"(small-mol={ (kept.ligand_class=='small-molecule').sum() }, "
          f"pep/prot={ (kept.ligand_class=='protein/peptide').sum() })")

    if limit:
        kept = kept.head(limit)
        print(f"    --limit -> processing metadata for first {len(kept)} kept structures")

    # Per-unique-receptor BW/TMH (residues/extended); cache once per protein
    proteins = sorted(p for p in kept.protein.dropna().unique())
    print(f"[1] fetching residue (BW/TMH) tables for {len(proteins)} unique receptors ...")
    residue_rows: list[dict] = []

    def fetch_residues(protein: str):
        data = _get_json(sess, f"{GPCRDB}/residues/extended/{protein}/")
        out = []
        for r in data or []:
            out.append({
                "protein": protein,
                "sequence_number": r.get("sequence_number"),
                "amino_acid": r.get("amino_acid"),
                "protein_segment": r.get("protein_segment"),  # TMH / loop id
                "bw_number": r.get("display_generic_number"),  # Ballesteros-Weinstein
            })
        return out

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(fetch_residues, proteins):
            residue_rows.extend(out)
    pd.DataFrame(residue_rows).to_parquet(META / "residues.parquet")
    print(f"    wrote residues.parquet ({len(residue_rows)} rows)")

    # Per-structure interaction (contact) tables
    pdbs = kept.pdb_code.tolist()
    print(f"[1] fetching interaction (contact) tables for {len(pdbs)} structures ...")
    contact_rows: list[dict] = []

    def fetch_interactions(pdb: str):
        data = _get_json(sess, f"{GPCRDB}/structure/{pdb}/interaction/")
        out = []
        for r in data or []:
            out.append({
                "pdb_code": pdb,
                "ligand_name": r.get("ligand_name"),
                "amino_acid": r.get("amino_acid"),
                "sequence_number": r.get("sequence_number"),
                "bw_number": r.get("display_generic_number"),
                "interaction_type": r.get("interaction_type"),
            })
        return out

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(fetch_interactions, pdbs):
            contact_rows.extend(out)
    pd.DataFrame(contact_rows).to_parquet(META / "contacts.parquet")
    print(f"    wrote contacts.parquet ({len(contact_rows)} rows)")
    print("[1] done.")


# --------------------------------------------------------------------------- #
# Step 2: download CIFs
# --------------------------------------------------------------------------- #
def stage_download(limit: int | None, workers: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(META / "structures.parquet")
    kept = df[df.keep].reset_index(drop=True)
    if limit:
        kept = kept.head(limit)
    pdbs = kept.pdb_code.tolist()
    sess = _session()
    failures: list[dict] = []

    def download_one(pdb: str):
        dest = RAW / f"{pdb}.cif"
        if dest.exists() and dest.stat().st_size > 0:
            return (pdb, "skip")
        url = RCSB_CIF.format(pdb=pdb)
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=120)
                if r.status_code == 200 and r.content:
                    dest.write_bytes(r.content)
                    return (pdb, "ok")
                if r.status_code == 404:
                    return (pdb, "404")
            except requests.RequestException:
                pass
            time.sleep(1.5 * (attempt + 1))
        return (pdb, "fail")

    print(f"[2] downloading {len(pdbs)} mmCIFs -> {RAW} ...")
    counts = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (pdb, status) in enumerate(ex.map(download_one, pdbs), 1):
            counts[status] += 1
            if status in ("404", "fail"):
                failures.append({"pdb_code": pdb, "status": status})
            if i % 100 == 0:
                print(f"    {i}/{len(pdbs)}  {counts}")
    print(f"[2] {counts}")
    if failures:
        pd.DataFrame(failures).to_csv(META / "download_failures.csv", index=False)
        print(f"    logged {len(failures)} failures -> meta/download_failures.csv")
    print("[2] done.")


# --------------------------------------------------------------------------- #
# Step 3: parse / clean / annotate
# --------------------------------------------------------------------------- #
def _load_and_clean(path: Path):
    from atomworks.io import parse
    out = parse(str(path), remove_waters=True, add_missing_atoms=True)
    aa = out["assemblies"]["1"][0]
    aa = aa[~np.isnan(aa.coord).any(axis=1)]
    return aa, out.get("chain_info", {})


def _annotate_interchain_contacts(aa, cutoff: float = CONTACT_CUTOFF):
    import biotite.structure as struc
    aa = aa.copy()
    cl = struc.CellList(aa.coord, cell_size=cutoff)
    mask = np.zeros(len(aa), dtype=bool)
    for ch in struc.get_chains(aa):
        this = aa.chain_id == ch
        near = np.any(cl.get_atoms(aa[this].coord, radius=cutoff, as_mask=True), axis=0)
        mask |= near & (~this)
    # store as int8 (0/1); booleans round-trip through CIF as ambiguous "True"/"False" strings
    aa.set_annotation("is_interchain_contact", mask.astype(np.int8))
    return aa


def _annotate_gpcrdb(aa, pdb: str, protein: str | None, residues_df, contacts_df):
    """Add per-atom bw_number / tmh segment / is_contact_residue from GPCRdb metadata.

    Joins on residue sequence number for the receptor's preferred chain(s). GPCRdb
    numbering is author-sequence based; we join on res_id which biotite exposes.
    """
    n = len(aa)
    # biotite annotations must be fixed-type arrays (object dtype is rejected)
    bw = np.full(n, "", dtype="U16")
    tmh = np.full(n, "", dtype="U16")
    is_contact = np.zeros(n, dtype=np.int8)  # int8 for clean CIF round-trip

    # BW / TMH per (sequence_number) for this receptor
    if protein is not None and residues_df is not None:
        rdf = residues_df[residues_df.protein == protein]
        seq_to_bw = dict(zip(rdf.sequence_number, rdf.bw_number.fillna("")))
        seq_to_seg = dict(zip(rdf.sequence_number, rdf.protein_segment.fillna("")))
    else:
        seq_to_bw = seq_to_seg = {}

    contact_seqnums = set()
    if contacts_df is not None:
        cdf = contacts_df[contacts_df.pdb_code == pdb]
        contact_seqnums = set(int(x) for x in cdf.sequence_number.dropna())

    res_id = aa.res_id
    for i in range(n):
        rid = int(res_id[i])
        if rid in seq_to_bw:
            bw[i] = seq_to_bw[rid] or ""
            tmh[i] = seq_to_seg[rid] or ""
        if rid in contact_seqnums:
            is_contact[i] = True

    aa.set_annotation("bw_number", bw)
    aa.set_annotation("tmh_segment", tmh)
    aa.set_annotation("is_contact_residue", is_contact)
    return aa


def stage_annotate(limit: int | None, workers: int) -> None:
    from atomworks.io.utils.io_utils import to_cif_file

    ANNOT.mkdir(parents=True, exist_ok=True)
    struct_df = pd.read_parquet(META / "structures.parquet")
    kept = struct_df[struct_df.keep].reset_index(drop=True)
    pdb_to_protein = dict(zip(kept.pdb_code, kept.protein))

    residues_df = pd.read_parquet(META / "residues.parquet") if (META / "residues.parquet").exists() else None
    contacts_df = pd.read_parquet(META / "contacts.parquet") if (META / "contacts.parquet").exists() else None

    cifs = sorted(RAW.glob("*.cif"))
    if limit:
        cifs = cifs[:limit]

    extra_fields = ["is_interchain_contact", "is_contact_residue", "bw_number", "tmh_segment"]
    print(f"[3] annotating {len(cifs)} structures -> {ANNOT} ...")

    def process(path: Path):
        pdb = path.stem.upper()
        dest = ANNOT / f"{pdb}.cif"
        if dest.exists() and dest.stat().st_size > 0:
            return (pdb, "skip")
        try:
            aa, _ = _load_and_clean(path)
            aa = _annotate_interchain_contacts(aa)
            aa = _annotate_gpcrdb(aa, pdb, pdb_to_protein.get(pdb), residues_df, contacts_df)
            to_cif_file(aa, str(dest), extra_fields=extra_fields)
            return (pdb, "ok")
        except Exception as e:  # noqa: BLE001
            return (pdb, f"ERR:{type(e).__name__}:{e}")

    counts = {"ok": 0, "skip": 0, "err": 0}
    errors = []
    # annotation is CPU-bound in atomworks; use processes only if safe -> threads keep it simple
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (pdb, status) in enumerate(ex.map(process, cifs), 1):
            if status in ("ok", "skip"):
                counts[status] += 1
            else:
                counts["err"] += 1
                errors.append({"pdb_code": pdb, "error": status})
            if i % 50 == 0:
                print(f"    {i}/{len(cifs)}  {counts}")
    print(f"[3] {counts}")
    if errors:
        pd.DataFrame(errors).to_csv(META / "annotate_failures.csv", index=False)
        print(f"    logged {len(errors)} errors -> meta/annotate_failures.csv")
    print("[3] done.")


# --------------------------------------------------------------------------- #
# Step 4: build manifest
# --------------------------------------------------------------------------- #
PEPTIDE_MAX_RESIDUES = 20  # AF3: a protein chain <= this counts as a peptide


def _manifest_row(pdb_and_raw: tuple[str, str]) -> dict | None:
    """Parse a raw CIF (fast, no missing-atom build) and compute composition.

    Runs in a worker process; returns a plain dict (picklable).
    """
    import numpy as _np
    from atomworks.io import parse as _parse

    pdb, raw_path = pdb_and_raw
    try:
        out = _parse(raw_path, add_missing_atoms=False, remove_waters=True)
    except Exception as e:  # noqa: BLE001
        return {"pdb_code": pdb, "error": f"{type(e).__name__}:{e}"}

    n_prot = n_pep = n_nuc = 0
    receptor_seq = ""
    for info in out["chain_info"].values():
        if not info.get("is_polymer"):
            continue
        ct = info.get("chain_type")
        ctn = (ct.name if hasattr(ct, "name") else str(ct)).upper()
        seq = info.get("processed_entity_canonical_sequence") or ""
        if "POLYPEPTIDE" in ctn or "PEPTIDE" in ctn:
            if len(seq) <= PEPTIDE_MAX_RESIDUES:
                n_pep += 1
            else:
                n_prot += 1
                if len(seq) > len(receptor_seq):
                    receptor_seq = seq  # longest protein chain = receptor
        elif "DNA" in ctn or "RNA" in ctn:
            n_nuc += 1

    aa = out["assemblies"]["1"][0]
    aa = aa[~_np.isnan(aa.coord).any(axis=1)]
    poly = _np.asarray(aa.is_polymer)
    n_lig = int(len(_np.unique(_np.asarray(aa.pn_unit_iid)[~poly]))) if (~poly).any() else 0

    return {
        "pdb_code": pdb,
        "n_prot": n_prot,
        "n_peptide": n_pep,
        "n_nuc": n_nuc,
        "n_ligand": n_lig,
        "total_num_atoms_in_unprocessed_assembly": int(len(aa)),
        "receptor_seq": receptor_seq,
    }


def stage_manifest(limit: int | None, workers: int) -> None:
    struct_df = pd.read_parquet(META / "structures.parquet").set_index("pdb_code")

    annotated = sorted(p.stem.upper() for p in ANNOT.glob("*.cif"))
    if limit:
        annotated = annotated[:limit]
    jobs = [(pdb, str(RAW / f"{pdb}.cif")) for pdb in annotated]
    print(f"[4] building manifest for {len(jobs)} annotated structures ...")

    rows, errors = [], []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_manifest_row, jobs), 1):
            if r is None or "error" in r:
                errors.append(r or {})
            else:
                rows.append(r)
            if i % 100 == 0:
                print(f"    {i}/{len(jobs)}  ok={len(rows)} err={len(errors)}")

    comp = pd.DataFrame(rows).set_index("pdb_code")
    keep_cols = ["protein", "receptor_class", "resolution", "method",
                 "publication_date", "ligand_class", "has_signalling_protein"]
    manifest = comp.join(struct_df[keep_cols], how="left").reset_index()
    manifest["example_id"] = manifest["pdb_code"]
    manifest["assembly_id"] = "1"
    manifest["path"] = manifest["pdb_code"].map(lambda p: str(ANNOT / f"{p}.cif"))
    manifest["deposition_date"] = manifest["publication_date"]  # proxy; GPCRdb has no deposition date

    manifest = manifest[[
        "example_id", "path", "assembly_id", "pdb_code", "protein", "receptor_class",
        "n_prot", "n_peptide", "n_nuc", "n_ligand",
        "total_num_atoms_in_unprocessed_assembly",
        "resolution", "method", "deposition_date", "ligand_class",
        "has_signalling_protein", "receptor_seq",
    ]]
    out_path = DATA / "gpcr_manifest.parquet"
    manifest.to_parquet(out_path)
    print(f"[4] wrote {out_path} ({len(manifest)} rows)")
    print(manifest.drop(columns=["receptor_seq"]).head(8).to_string(index=False))
    if errors:
        pd.DataFrame(errors).to_csv(META / "manifest_failures.csv", index=False)
        print(f"    logged {len(errors)} parse failures -> meta/manifest_failures.csv")
    print("[4] done.")


# --------------------------------------------------------------------------- #
# Step 5: cluster + train/val split
# --------------------------------------------------------------------------- #
def stage_split(val_fraction: float, seed: int) -> None:
    """Cluster by GPCRdb receptor identity and split whole clusters into train/val.

    mmseqs is unavailable here, and receptor sequences are contaminated by
    crystallization fusion partners (e.g. T4 lysozyme), so we cluster on the
    curated GPCRdb `protein` id with the species suffix removed (adrb1_human,
    adrb1_melga -> adrb1). This groups orthologs, preventing species leakage.
    Split is stratified by (receptor_class, dominant ligand class) so val
    contains both small-molecule and peptide/protein complexes AND covers the
    receptor classes: for any stratum with >=2 clusters we guarantee at least
    one held-out cluster.
    """
    m = pd.read_parquet(DATA / "gpcr_manifest.parquet")
    # receptor identity = protein name minus the trailing species token
    m["cluster"] = m["protein"].str.rsplit("_", n=1).str[0]

    size = m.groupby("cluster").size()
    dom_lig = m.groupby("cluster")["ligand_class"].agg(lambda s: s.value_counts().idxmax())
    rec_class = m.groupby("cluster")["receptor_class"].agg(lambda s: s.value_counts().idxmax())
    cl = pd.DataFrame({"size": size, "ligand": dom_lig, "receptor_class": rec_class})
    cl["stratum"] = list(zip(cl["receptor_class"], cl["ligand"]))

    rng = np.random.default_rng(seed)
    val_clusters: set[str] = set()
    for stratum, grp in cl.groupby("stratum"):
        n = len(grp)
        if n < 2:
            continue  # single receptor cluster: keep in train (can't hold out without losing it)
        target = val_fraction * grp["size"].sum()
        order = rng.permutation(grp.index.to_numpy())
        added: list[str] = []
        acc = 0
        for c in order:
            if len(added) >= n - 1:
                break  # always leave >=1 cluster in train
            if acc >= target and added:
                break
            csize = int(cl.loc[c, "size"])
            if csize > target:
                continue  # never let a single oversized cluster represent the stratum's val
            added.append(c)
            acc += csize
        # guarantee coverage: if every cluster was oversized, hold out the smallest
        if not added:
            added.append(grp["size"].idxmin())
        val_clusters.update(added)

    m["split"] = np.where(m["cluster"].isin(val_clusters), "val", "train")
    m.to_parquet(DATA / "gpcr_manifest.parquet")  # persist cluster + split back

    train = m[m.split == "train"].reset_index(drop=True)
    val = m[m.split == "val"].reset_index(drop=True)
    train.to_parquet(DATA / "gpcr_manifest_train.parquet")
    val.to_parquet(DATA / "gpcr_manifest_val.parquet")

    assert set(train.cluster) & set(val.cluster) == set(), "cluster leakage between splits!"

    print(f"[5] clusters={m.cluster.nunique()}  seed={seed}  val_fraction={val_fraction}")
    print(f"    train: {len(train)} structures / {train.cluster.nunique()} clusters")
    print(f"    val:   {len(val)} structures / {val.cluster.nunique()} clusters "
          f"({len(val)/len(m):.1%} of data)")
    print("    val ligand_class:", val.ligand_class.value_counts().to_dict())
    print("    val receptor_class:", val.receptor_class.value_counts().to_dict())
    print("    disjoint clusters: OK")
    print("[5] done.")


# --------------------------------------------------------------------------- #
# Step 6a: MSA prep (local) - extract unique protein sequences -> FASTA for HPC
# --------------------------------------------------------------------------- #
def _protein_chain_seqs(pdb: str) -> list[str]:
    from atomworks.io import parse as _parse
    try:
        out = _parse(f"{RAW}/{pdb}.cif", add_missing_atoms=False, remove_waters=True)
    except Exception:
        return []
    seqs = []
    for info in out["chain_info"].values():
        if not info.get("is_polymer"):
            continue
        ct = info.get("chain_type")
        ctn = (ct.name if hasattr(ct, "name") else str(ct)).upper()
        if "POLYPEPTIDE" in ctn:
            s = info.get("processed_entity_canonical_sequence") or ""
            if s:
                seqs.append(s)
    return seqs


def stage_msa_prep(min_len: int, workers: int) -> None:
    """Extract unique protein chain sequences and write a FASTA (headers = 11-char
    sequence hash) for MSA generation on an HPC. Also writes a seq index TSV."""
    from atomworks.ml.utils.misc import hash_sequence

    out_dir = DATA / "msa_input"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdbs = pd.read_parquet(DATA / "gpcr_manifest.parquet").pdb_code.tolist()

    seqs: set[str] = set()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for chain_seqs in ex.map(_protein_chain_seqs, pdbs):
            seqs.update(chain_seqs)

    kept = sorted(s for s in seqs if len(s) >= min_len)
    skipped = len(seqs) - len(kept)

    fasta = out_dir / "queries.fasta"
    index = out_dir / "seq_index.tsv"
    seen_hashes: dict[str, str] = {}
    with open(fasta, "w") as ff, open(index, "w") as fi:
        fi.write("hash\tlength\tsequence\n")
        for s in kept:
            h = hash_sequence(s)
            if h in seen_hashes:  # 11-char hash collision guard (astronomically unlikely)
                continue
            seen_hashes[h] = s
            ff.write(f">{h}\n{s}\n")
            fi.write(f"{h}\t{len(s)}\t{s}\n")

    print(f"[6a] unique protein seqs: {len(seqs)} | kept (len>={min_len}): {len(kept)} | "
          f"skipped short: {skipped}")
    print(f"     wrote {fasta} ({len(seen_hashes)} records) and {index}")
    print(f"     -> copy {out_dir} to the HPC and run hpc_generate_msas.sh")
    print("[6a] done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch-meta", "download", "annotate", "manifest",
                                      "split", "msa-prep"])
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-len", type=int, default=16, help="min protein length to request an MSA")
    ap.add_argument("--limit", type=int, default=None, help="process only first N kept structures")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.stage == "fetch-meta":
        stage_fetch_meta(args.limit, args.workers)
    elif args.stage == "download":
        stage_download(args.limit, args.workers)
    elif args.stage == "annotate":
        stage_annotate(args.limit, args.workers)
    elif args.stage == "manifest":
        stage_manifest(args.limit, args.workers)
    elif args.stage == "split":
        stage_split(args.val_fraction, args.seed)
    elif args.stage == "msa-prep":
        stage_msa_prep(args.min_len, args.workers)


if __name__ == "__main__":
    main()
