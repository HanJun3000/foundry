"""Boltz wrapper used as the structure-prediction step of the consensus
cycling pipeline.

It converts an AF3-style JSON spec into a Boltz YAML, runs Boltz via the
Apptainer image, locates the predicted CIF / confidence / PAE files, and
returns metrics in a stable AF3-like shape that the cycling driver consumes.

All heavy assets (the Boltz ``.sif`` container and the model checkpoint) are
*not* shipped with this repository.  They are configured through environment
variables (see ``README.md``):

    BOLTZ_EXECUTABLE      path to a separately installed boltz (default: boltz)
    BOLTZ_SIF             optional path to the Boltz apptainer image
    BOLTZ_CKPT            path to the Boltz checkpoint (e.g. boltz2_conf.ckpt)
    BOLTZ_CACHE           Boltz cache directory (CCD / molecule cache)
    BOLTZ_BIND_PATHS      comma-separated host paths to bind into the container
    BOLTZ_RECYCLING_STEPS (default 3)
    BOLTZ_SAMPLING_STEPS  (default 200)
    BOLTZ_NUM_WORKERS     (default 2)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from Bio.PDB import MMCIFParser, PDBParser, Superimposer

# ============================================================
# Boltz environment configuration (all overridable via env vars)
# ============================================================
BOLTZ_SIF = os.environ.get("BOLTZ_SIF", "")
BOLTZ_EXECUTABLE = os.environ.get("BOLTZ_EXECUTABLE", "boltz")
BOLTZ_CKPT = os.environ.get("BOLTZ_CKPT", "")
BOLTZ_CACHE = os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.cache/boltz"))
BOLTZ_RECYCLING_STEPS = int(os.environ.get("BOLTZ_RECYCLING_STEPS", "3"))
BOLTZ_SAMPLING_STEPS = int(os.environ.get("BOLTZ_SAMPLING_STEPS", "200"))
BOLTZ_NUM_WORKERS = int(os.environ.get("BOLTZ_NUM_WORKERS", "2"))
BOLTZ_BIND_PATHS = os.environ.get("BOLTZ_BIND_PATHS", "")


# ============================================================
# AF3 JSON -> Boltz YAML conversion
# ============================================================
def af3_json_to_boltz_yaml(
    af3_json_path: str,
    boltz_yaml_path: str,
    cyclic_chains: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Convert an AF3-style JSON file (single-entry list) into a Boltz YAML.

    Returns (boltz_yaml_path, name).
    """
    with open(af3_json_path, "r") as f:
        af3_data = json.load(f)
    entry = af3_data[0] if isinstance(af3_data, list) else af3_data

    name = entry.get("name", Path(af3_json_path).stem)

    boltz_sequences: List[Dict] = []
    for seq_block in entry.get("sequences", []):
        if "protein" in seq_block:
            prot = seq_block["protein"]
            chain_id = prot["id"]
            sequence = prot.get("sequence", "")
            entry_dict: Dict = {"id": chain_id, "sequence": sequence}

            msa_path = prot.get("unpairedMsaPath") or prot.get("msaPath")
            unpaired_msa_text = prot.get("unpairedMsa")
            if msa_path:
                msa_file = Path(msa_path).expanduser().resolve()
                if not msa_file.is_file():
                    raise FileNotFoundError(f"MSA does not exist: {msa_file}")
                entry_dict["msa"] = str(msa_file)
            elif unpaired_msa_text:
                msa_out = Path(boltz_yaml_path).parent / f"{name}_{chain_id}.a3m"
                msa_out.write_text(unpaired_msa_text)
                entry_dict["msa"] = str(msa_out)
            else:
                # Force single-sequence mode (Boltz requires an MSA otherwise).
                entry_dict["msa"] = "empty"

            if cyclic_chains and chain_id in cyclic_chains:
                entry_dict["cyclic"] = True

            boltz_sequences.append({"protein": entry_dict})
        elif "dna" in seq_block or "rna" in seq_block:
            kind = "dna" if "dna" in seq_block else "rna"
            polymer = seq_block[kind]
            boltz_sequences.append(
                {kind: {"id": polymer["id"], "sequence": polymer["sequence"]}}
            )
        elif "ligand" in seq_block:
            lig = seq_block["ligand"]
            entry_dict = {"id": lig["id"]}
            if "smiles" in lig:
                entry_dict["smiles"] = lig["smiles"]
            elif "ccd" in lig:
                entry_dict["ccd"] = lig["ccd"]
            elif len(lig.get("ccdCodes", [])) == 1:
                entry_dict["ccd"] = lig["ccdCodes"][0]
            else:
                for key in ("SMILES", "smile", "smi"):
                    if key in lig:
                        entry_dict["smiles"] = lig[key]
                        break
            boltz_sequences.append({"ligand": entry_dict})
        else:
            raise ValueError(f"Unsupported AF3 entity: {list(seq_block)}")

    boltz_doc = {"version": 1, "sequences": boltz_sequences}

    Path(boltz_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(boltz_yaml_path, "w") as f:
        yaml.safe_dump(boltz_doc, f, sort_keys=False)
    return boltz_yaml_path, name


# ============================================================
# Boltz invocation
# ============================================================
def run_boltz(
    boltz_yaml_path: str,
    output_dir: str,
    num_samples: int = 1,
    recycling_steps: Optional[int] = None,
    sampling_steps: Optional[int] = None,
    use_potentials: bool = False,
    output_format: str = "mmcif",
    description: str = "Boltz Inference",
) -> Tuple[bool, float]:
    """Run Boltz prediction on a YAML file.  Returns (success, wall_time)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(BOLTZ_CACHE).mkdir(parents=True, exist_ok=True)
    recycling = (
        recycling_steps if recycling_steps is not None else BOLTZ_RECYCLING_STEPS
    )
    sampling = sampling_steps if sampling_steps is not None else BOLTZ_SAMPLING_STEPS

    cmd_parts = []
    if BOLTZ_SIF:
        paths = [
            str(Path(boltz_yaml_path).resolve().parent),
            str(Path(output_dir).resolve()),
            str(Path(BOLTZ_CACHE).resolve()),
        ]
        if BOLTZ_CKPT:
            paths.append(str(Path(BOLTZ_CKPT).resolve().parent))
        # Include any external per-chain MSA directories.
        with open(boltz_yaml_path) as f:
            for block in yaml.safe_load(f).get("sequences", []):
                msa = block.get("protein", {}).get("msa")
                if msa and msa != "empty":
                    paths.append(str(Path(msa).resolve().parent))
        if BOLTZ_BIND_PATHS:
            paths.append(BOLTZ_BIND_PATHS)
        cmd_parts = [
            "apptainer",
            "exec",
            "--nv",
            "-B",
            ",".join(dict.fromkeys(paths)),
            BOLTZ_SIF,
        ]
    cmd_parts += [
        BOLTZ_EXECUTABLE,
        "predict",
        boltz_yaml_path,
        "--out_dir",
        output_dir,
        "--cache",
        BOLTZ_CACHE,
        "--model",
        "boltz2",
    ]
    if BOLTZ_CKPT:
        cmd_parts.extend(["--checkpoint", BOLTZ_CKPT])
    cmd_parts.extend(
        [
            "--recycling_steps",
            str(recycling),
            "--sampling_steps",
            str(sampling),
            "--diffusion_samples",
            str(num_samples),
            "--output_format",
            output_format,
            "--num_workers",
            str(BOLTZ_NUM_WORKERS),
            "--no_kernels",
            "--override",
        ]
    )
    if use_potentials:
        cmd_parts.append("--use_potentials")

    cmd_str = shlex.join(cmd_parts)
    print(f"\nBoltz [{description}] cmd:\n{cmd_str}\n")
    start = time.time()
    try:
        subprocess.run(cmd_parts, check=True)
        return True, time.time() - start
    except subprocess.CalledProcessError as e:
        print(f"Boltz failed (rc={e.returncode}) on {boltz_yaml_path}")
        return False, time.time() - start


# ============================================================
# Output discovery + parsing
# ============================================================
def find_boltz_outputs(out_dir: str, name: str) -> Dict[str, Optional[str]]:
    """Locate the canonical files produced by ``boltz predict``.

    Boltz writes them to ``out_dir/boltz_results_<name>/predictions/<name>/``.
    """
    base = Path(out_dir) / f"boltz_results_{name}" / "predictions" / name
    cif = base / f"{name}_model_0.cif"
    confidence = base / f"confidence_{name}_model_0.json"
    pae = base / f"pae_{name}_model_0.npz"
    plddt = base / f"plddt_{name}_model_0.npz"

    return {
        "base": str(base),
        "cif": str(cif) if cif.exists() else None,
        "confidence": str(confidence) if confidence.exists() else None,
        "pae": str(pae) if pae.exists() else None,
        "plddt": str(plddt) if plddt.exists() else None,
    }


def _calculate_average_b_factor(cif_path: str) -> float:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("p", cif_path)
    bs = [
        atom.bfactor
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    return sum(bs) / len(bs) if bs else 0.0


def _get_parser(filename: str):
    if filename.endswith(".cif"):
        return MMCIFParser(QUIET=True)
    if filename.endswith(".pdb"):
        return PDBParser(QUIET=True)
    raise ValueError(f"Unsupported format: {filename}")


def _sorted_ca_atoms(model):
    atoms = []
    for chain in sorted(model, key=lambda c: c.id):
        for residue in sorted(chain, key=lambda r: r.id[1]):
            if residue.has_id("CA"):
                atoms.append(residue["CA"])
    return atoms


def _calculate_ca_rmsd(first_file: str, second_file: str) -> Optional[float]:
    try:
        s1 = _get_parser(first_file).get_structure("ref", first_file)
        s2 = _get_parser(second_file).get_structure("mob", second_file)
        m1 = next(s1.get_models())
        m2 = next(s2.get_models())
        a1 = _sorted_ca_atoms(m1)
        a2 = _sorted_ca_atoms(m2)
        if len(a1) != len(a2):
            return None
        si = Superimposer()
        si.set_atoms(a1, a2)
        si.apply(a2)
        return si.rms
    except Exception as e:
        print(f"rmsd failed: {e}")
        return None


def _chain_token_ranges(cif_path: str) -> Dict[str, Tuple[int, int]]:
    """Approximate per-chain token (residue/atom) ranges from a Boltz CIF."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("p", cif_path)
    model = next(structure.get_models())

    standard_aa = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "MSE",
        "SEC",
        "PYL",
        "UNK",
    }
    standard_na = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "N"}

    ranges: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for chain in model:
        n_tokens = 0
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname in standard_aa or resname in standard_na:
                n_tokens += 1
            else:
                n_tokens += sum(1 for _ in residue)
        ranges[chain.id] = (cursor, cursor + n_tokens)
        cursor += n_tokens
    return ranges


def _interface_pae(
    pae_path: str, cif_path: str, chain_a: str = "A", chain_b: str = "B"
) -> Tuple[float, float]:
    """Return (mean_interface_pae, min_interface_pae) between two chains."""
    try:
        data = np.load(pae_path)
        pae = data[list(data.keys())[0]]
        ranges = _chain_token_ranges(cif_path)
        if chain_a not in ranges or chain_b not in ranges:
            return float("nan"), float("nan")
        a0, a1 = ranges[chain_a]
        b0, b1 = ranges[chain_b]
        if a1 <= a0 or b1 <= b0 or a1 > pae.shape[0] or b1 > pae.shape[1]:
            return float("nan"), float("nan")
        block_ab = pae[a0:a1, b0:b1]
        block_ba = pae[b0:b1, a0:a1]
        mean_pae = float((block_ab.mean() + block_ba.mean()) / 2)
        min_pae = float((block_ab.min() + block_ba.min()) / 2)
        return mean_pae, min_pae
    except Exception as e:
        print(f"interface PAE failed: {e}")
        return float("nan"), float("nan")


def parse_boltz_metrics(
    confidence_path: Optional[str],
    cif_path: Optional[str],
    pae_path: Optional[str],
    copied_pdb: Optional[str],
    chain_a: str = "A",
    chain_b: str = "B",
) -> Dict:
    """Build an AF3-shaped metrics dict from Boltz outputs."""
    out: Dict = {
        "AF3_PTM": np.nan,
        "AF3_iPTM": np.nan,
        "AF3_iPAE": np.nan,
        "AF3_design_ptm": np.nan,
        "AF3_mean_pair_iptm": np.nan,
        "AF3_pLDDT": np.nan,
        "AF3_rmsd": np.nan,
        "ranking_score": np.nan,
        "AF3_Status": "Failed",
    }

    if not confidence_path or not cif_path:
        return out

    try:
        with open(confidence_path) as f:
            conf = json.load(f)

        out["AF3_PTM"] = conf.get("ptm", np.nan)
        out["AF3_iPTM"] = conf.get("iptm", np.nan)
        out["ranking_score"] = conf.get("confidence_score", conf.get("ptm", np.nan))

        chains_ptm = conf.get("chains_ptm") or {}
        if chains_ptm:
            first_key = sorted(chains_ptm.keys())[0]
            out["AF3_design_ptm"] = chains_ptm[first_key]

        pair = conf.get("pair_chains_iptm") or {}
        if (
            pair
            and "0" in pair
            and "1" in pair.get("0", {})
            and "0" in pair.get("1", {})
        ):
            out["AF3_mean_pair_iptm"] = (
                float(pair["0"]["1"]) + float(pair["1"]["0"])
            ) / 2.0

        plddt_val = conf.get("complex_plddt")
        if plddt_val is not None:
            # Boltz reports complex_plddt in [0, 1]; scale to 0..100.
            out["AF3_pLDDT"] = float(plddt_val) * 100.0
        else:
            out["AF3_pLDDT"] = _calculate_average_b_factor(cif_path)

        if pae_path and Path(pae_path).exists():
            _, min_pae = _interface_pae(pae_path, cif_path, chain_a, chain_b)
            out["AF3_iPAE"] = min_pae

        if copied_pdb and Path(copied_pdb).exists():
            out["AF3_rmsd"] = _calculate_ca_rmsd(cif_path, copied_pdb)

        out["AF3_Status"] = "Success"
    except Exception as e:
        print(f"parse_boltz_metrics failed: {e}")

    return out
