#!/usr/bin/env python
"""Cycling driver. Each cycle:

    1. With probability ``backprop_fraction`` run the RF3 backprop optimizer
       (``backprop/optimize.py``); otherwise run Boltz on the current sequence.
       Both produce a structure for the current design.
    2. Redesign the binder chain on that structure with MPNN.
    3. The redesigned structure becomes the input to the next cycle.

Each model runs in a separate process (optionally Apptainer). The driver only
needs biopython / pyyaml / numpy; see ``docs/installation.md`` for setup.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from Bio.PDB import MMCIFParser, PDBParser

from rfo.cycling.boltz.boltz_runner import (
    af3_json_to_boltz_yaml,
    find_boltz_outputs,
    parse_boltz_metrics,
    run_boltz,
)

# Standard 3-letter -> 1-letter map (hardcoded to avoid biopython-version drift;
# older Bio.PDB.Polypeptide lacks ``protein_letters_3to1``).
protein_letters_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


# ============================================================
# Configuration (env-overridable; documented in cycling/README.md)
# ============================================================
RFO_SRC = Path(__file__).resolve().parents[2]
BACKPROP_DIR = RFO_SRC / "rfo" / "backprop"
# Source checkout and wheel installs both work. Only add neighboring Foundry
# source trees if this is actually a checkout.
_source_paths = [str(RFO_SRC)]
_repo = RFO_SRC.parent.parent.parent
if (_repo / "models" / "rf3" / "src").is_dir():
    _source_paths += [str(_repo / "src"), str(_repo / "models" / "rf3" / "src")]
RF3_CONTAINER_PYTHONPATH = os.environ.get(
    "RF3_CONTAINER_PYTHONPATH", os.pathsep.join(_source_paths)
)
RF3_SIF = os.environ.get("RF3_SIF", "")
RF3_PYTHON = os.environ.get("RF3_PYTHON") or ("python" if RF3_SIF else sys.executable)
RF3_BIND_PATHS = os.environ.get("RF3_BIND_PATHS", "")
_FORWARD_ENV = {
    key: os.environ.get(key, "")
    for key in ("CCD_MIRROR_PATH", "PDB_MIRROR_PATH", "X3DNA", "DSSP", "RF3_CKPT")
}


def rf3_command(config_file: Path, bind_paths=()) -> list:
    """Run an output-local config using this RFO installation."""
    cmd = [
        RF3_PYTHON,
        str(BACKPROP_DIR / "optimize.py"),
        "--config_name",
        config_file.stem,
        "--config_path",
        str(config_file.parent.resolve()),
    ]
    if not RF3_SIF:
        return cmd
    paths = [*_source_paths, str(config_file.parent.resolve()), *bind_paths]
    paths += [
        str(Path(v).expanduser().resolve().parent) for v in _FORWARD_ENV.values() if v
    ]
    if RF3_BIND_PATHS:
        paths.append(RF3_BIND_PATHS)
    prefix = [
        "apptainer",
        "exec",
        "--nv",
        "-B",
        ",".join(dict.fromkeys(paths)),
        "--env",
        f"PYTHONPATH={RF3_CONTAINER_PYTHONPATH}",
    ]
    for key, value in _FORWARD_ENV.items():
        if value:
            prefix += ["--env", f"{key}={value}"]
    return [*prefix, RF3_SIF, *cmd]


# ============================================================
# Small structure helpers (pure Bio/biotite — no torch needed)
# ============================================================
def _structure_parser(path: str):
    return MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)


def get_chain_sequence(path: str, chain_id: str = "A") -> str:
    """Return the one-letter sequence of ``chain_id`` from a PDB/CIF file."""
    structure = _structure_parser(path).get_structure("s", path)
    model = next(structure.get_models())
    seq = []
    if chain_id in [c.id for c in model]:
        chain = model[chain_id]
    else:
        raise ValueError(f"Required chain {chain_id!r} is missing from {path}")
    for residue in chain:
        if residue.id[0] != " ":
            continue  # skip hetero/water
        resname = residue.get_resname().strip().upper()
        seq.append(protein_letters_3to1.get(resname, "X"))
    return "".join(seq)


def structure_to_af3_sequences(path: str) -> List[Dict]:
    """Build protein, nucleic-acid or CCD ligand blocks from distinct chains."""
    structure = _structure_parser(path).get_structure("s", path)
    blocks = []
    alphabets = {
        "protein": protein_letters_3to1,
        "dna": {"DA": "A", "DC": "C", "DG": "G", "DT": "T"},
        "rna": {"A": "A", "C": "C", "G": "G", "U": "U"},
    }
    for chain in next(structure.get_models()):
        residues = [r for r in chain if r.resname.strip() not in {"HOH", "WAT"}]
        if not residues:
            continue
        names = [r.resname.strip().upper() for r in residues]
        for kind, alphabet in alphabets.items():
            if all(name in alphabet for name in names):
                blocks.append(
                    {
                        kind: {
                            "id": chain.id,
                            "sequence": "".join(alphabet[name] for name in names),
                        }
                    }
                )
                break
        else:
            if len(residues) != 1:
                raise ValueError(
                    f"Cannot infer a single entity for chain {chain.id}; use "
                    "af3_json_template and separate chain IDs for ligand entities."
                )
            blocks.append({"ligand": {"id": chain.id, "ccd": names[0]}})
    return blocks


def force_copy(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(dst):
        try:
            os.chmod(dst, 0o777)
            os.remove(dst)
        except OSError:
            pass
    shutil.copy2(src, dst)


def cif_to_pdb(cif_path: str, pdb_path: str) -> bool:
    """Convert a CIF to PDB, truncating >3-char residue names for PDB rules."""
    try:
        import biotite.structure.io.pdb as pdb
        import biotite.structure.io.pdbx as pdbx

        cif = pdbx.CIFFile.read(cif_path)
        atom_array = pdbx.get_structure(cif, model=1)
        # PDB res_name field is 3 chars max.
        fixed = atom_array.res_name.copy()
        for i, name in enumerate(fixed):
            if len(name) > 3:
                fixed[i] = name[:3]
        atom_array.res_name = fixed
        out = pdb.PDBFile()
        pdb.set_structure(out, atom_array)
        out.write(pdb_path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"cif_to_pdb failed ({e}); trying Bio.PDB fallback")
        try:
            from Bio.PDB import PDBIO, MMCIFParser

            s = MMCIFParser(QUIET=True).get_structure("s", cif_path)
            io = PDBIO()
            io.set_structure(s)
            io.save(pdb_path)
            return True
        except Exception as e2:  # noqa: BLE001
            print(f"Bio.PDB fallback failed: {e2}")
            return False


def get_residues_within_cutoff(
    path: str, chain_prot: str = "A", chain_lig: str = "B", cutoff: float = 8.0
) -> List[int]:
    """1-indexed residues of ``chain_prot`` with any atom within ``cutoff`` Å
    of ``chain_lig`` (used to fix interface residues during redesign)."""
    structure = _structure_parser(path).get_structure("s", path)
    model = next(structure.get_models())
    chains = {c.id: c for c in model}
    if chain_prot not in chains or chain_lig not in chains:
        return []
    lig_atoms = [a for r in chains[chain_lig] for a in r]
    contacts = []
    for i, residue in enumerate(
        (r for r in chains[chain_prot] if r.id[0] == " "), start=1
    ):
        hit = any((a - la) <= cutoff for a in residue for la in lig_atoms)
        if hit:
            contacts.append(i)
    return sorted(contacts)


def template_process(
    cif_path: str, plddt_threshold: float = 80, min_continuous_length: int = 5
) -> List[int]:
    """Return 0-indexed chain-A residues whose mean pLDDT (CIF B-factor) is
    above ``plddt_threshold`` and that fall in a contiguous run of at least
    ``min_continuous_length`` residues."""
    structure = MMCIFParser(QUIET=True).get_structure("confidence", cif_path)
    model = next(structure.get_models())
    if "A" not in model:
        raise ValueError("Boltz output is missing designed chain A.")
    residues = [r for r in model["A"] if r.id[0] == " "]
    # Masks use positions in chain A, not PDB author residue numbers.
    good = [
        i
        for i, res in enumerate(residues)
        if sum(atom.bfactor for atom in res) / len(res) > plddt_threshold
    ]

    # keep contiguous runs only
    out: List[int] = []
    run: List[int] = []
    for rid in good:
        if run and rid == run[-1] + 1:
            run.append(rid)
        else:
            if len(run) >= min_continuous_length:
                out.extend(run)
            run = [rid]
    if len(run) >= min_continuous_length:
        out.extend(run)
    return out


def process_history_metrics(history_path: str) -> Tuple[Dict, Optional[int]]:
    """Parse the optimizer ``history.json`` and return (metrics, best_step).

    ``best_step`` is the ``step`` field of the lowest-loss entry (used to find
    ``modelhub_pred/<step>.cif``)."""
    metrics: Dict = {"RF3_Status": "Failed"}
    try:
        with open(history_path) as f:
            history = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"cannot read history {history_path}: {e}")
        return metrics, None
    if not history:
        return metrics, None

    def loss_of(entry):
        for k in ("loss_prop", "loss"):
            if k in entry and entry[k] is not None:
                return entry[k]
        return float("inf")

    best = min(history, key=loss_of)
    metrics = {
        "RF3_Status": "Success",
        "step": best.get("step"),
        "loss": loss_of(best),
        "ipae_min": best.get("ipae_min", float("nan")),
        "ipae_mean": best.get("ipae_mean", float("nan")),
        "iptm": best.get("iptm", float("nan")),
        "substrate_plddt": best.get("substrate_plddt", float("nan")),
    }
    return metrics, best.get("step")


# ============================================================
# Branch: Boltz structure prediction
# ============================================================
def run_boltz_branch(
    af3_json_template: str,
    structure_file: str,
    cycle: int,
    output_dir: str,
    num_samples: int,
    template_plddt_threshold: int,
    cyclic_chains: Optional[List[str]],
    msa_paths: Optional[Dict] = None,
) -> Tuple[Dict, Optional[str], List[int]]:
    """Predict the current design with Boltz. Returns (metrics, pdb_path,
    fixed_residues_0idx)."""
    target_dir = os.path.join(output_dir, f"recycle_{cycle + 1}")
    os.makedirs(target_dir, exist_ok=True)

    stem = Path(structure_file).stem
    tag = f"{stem}_recycle_{cycle + 1}"
    copied_pdb = os.path.join(target_dir, f"{tag}.pdb")
    force_copy(structure_file, copied_pdb)

    # Build the AF3 json for this cycle. By default we read every chain
    # straight from the current structure (so chain A is the current design);
    # if an explicit template is given we use it and only refresh chain A's
    # sequence (lets you pin ligands-via-SMILES / embedded MSAs).
    if af3_json_template:
        with open(af3_json_template) as f:
            af3 = copy.deepcopy(json.load(f))
        entry = af3[0] if isinstance(af3, list) else af3
        entry["name"] = tag
        binder = [
            block["protein"]
            for block in entry["sequences"]
            if "protein" in block and block["protein"].get("id") in ("A", ["A"])
        ]
        if len(binder) != 1:
            raise ValueError(
                "af3_json_template must contain one protein entry for chain A."
            )
        binder[0]["sequence"] = get_chain_sequence(copied_pdb, "A")
    else:
        entry = {"name": tag, "sequences": structure_to_af3_sequences(copied_pdb)}
    if msa_paths:
        for block in entry.get("sequences", []):
            prot = block.get("protein")
            if prot and prot.get("id") in msa_paths:
                prot["unpairedMsaPath"] = msa_paths[prot["id"]]
    json_path = os.path.join(target_dir, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump([entry], f, indent=2)

    yaml_path = os.path.join(target_dir, f"{tag}.yaml")
    af3_json_to_boltz_yaml(json_path, yaml_path, cyclic_chains=cyclic_chains)
    success, cost = run_boltz(yaml_path, target_dir, num_samples=num_samples)

    metrics = {"method": "Boltz", "Inference_Cost_Sec": cost}
    if not success:
        metrics["AF3_Status"] = "Failed"
        return metrics, None, []

    paths = find_boltz_outputs(target_dir, tag)
    if not paths["cif"] or not paths["confidence"]:
        print(f"Boltz outputs missing: {paths}")
        metrics["AF3_Status"] = "Failed"
        return metrics, None, []

    metrics.update(
        parse_boltz_metrics(paths["confidence"], paths["cif"], paths["pae"], copied_pdb)
    )
    cif_out = os.path.join(target_dir, f"{tag}.cif")
    shutil.copy2(paths["cif"], cif_out)
    pdb_out = cif_out.replace(".cif", ".pdb")
    if not cif_to_pdb(cif_out, pdb_out):
        return metrics, None, []

    fixed = template_process(paths["cif"], template_plddt_threshold)
    return metrics, pdb_out, fixed


# ============================================================
# Branch: RF3 gradient/MCMC sequence optimization (apptainer)
# ============================================================
def run_rf3_branch(
    base_config: str,
    structure_file: str,
    cycle: int,
    output_dir: str,
    cyclic: bool,
    loss: Optional[str] = None,
    msa_paths: Optional[Dict] = None,
    templates: Optional[Dict] = None,
) -> Tuple[Dict, Optional[str], List[int]]:
    """Optimize the current design with the backprop RF3 optimizer. Returns
    (metrics, pdb_path, fixed_residues_0idx)."""
    target_dir = os.path.join(output_dir, f"recycle_{cycle + 1}")
    os.makedirs(target_dir, exist_ok=True)
    stem = Path(structure_file).stem

    # Generate a per-cycle config from the user template: override input +
    # output_path (and any optional loss / msa / template) so the optimizer
    # reads our structure and writes here.
    with open(base_config) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    cfg["input"] = os.path.abspath(structure_file)
    cfg["output_path"] = os.path.abspath(target_dir)
    if loss:
        cfg["loss_type"] = loss
    if msa_paths:
        cfg["msa_paths"] = msa_paths
    if templates:
        cfg["fix_template_dict"] = templates

    # Per-cycle files belong to the run, not a potentially read-only install.
    config_file = Path(target_dir).resolve() / "rf3_config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    paths = [
        str(Path(structure_file).resolve().parent),
        str(Path(output_dir).resolve()),
    ]
    for mapping in (
        msa_paths,
        templates,
        cfg.get("msa_paths"),
        cfg.get("fix_template_dict"),
    ):
        if isinstance(mapping, dict):
            paths.extend(
                str(Path(v).expanduser().resolve().parent) for v in mapping.values()
            )
    checkpoints = cfg.get("checkpoint_paths") or cfg.get("checkpoint_path", [])
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]
    paths.extend(
        str(Path(v).expanduser().resolve().parent) for v in checkpoints if "${" not in v
    )
    cmd = rf3_command(config_file, paths)
    env = os.environ.copy()
    env["PYTHONPATH"] = RF3_CONTAINER_PYTHONPATH
    print("\nRF3 optimizer cmd:\n" + " ".join(cmd) + "\n")
    start = time.time()
    try:
        subprocess.run(cmd, env=env, check=True)
        success, cost = True, time.time() - start
    except subprocess.CalledProcessError as e:
        print(f"RF3 optimizer failed (rc={e.returncode})")
        success, cost = False, time.time() - start

    metrics = {"method": "RF3", "Inference_Cost_Sec": cost}
    if not success:
        metrics["RF3_Status"] = "Failed"
        return metrics, None, []

    # The optimizer writes to <output_path>/<stem><suffix>/ — find it.
    hist_candidates = sorted(
        glob.glob(os.path.join(target_dir, "*", "history.json")),
        key=os.path.getmtime,
    )
    if not hist_candidates:
        print(f"RF3: no history.json under {target_dir}")
        metrics["RF3_Status"] = "Failed"
        return metrics, None, []
    opt_dir = os.path.dirname(hist_candidates[-1])

    rf3_metrics, best_step = process_history_metrics(hist_candidates[-1])
    metrics.update(rf3_metrics)

    # Locate the predicted structure for the best step.
    pred_dir = os.path.join(opt_dir, "modelhub_pred")
    cif_src = os.path.join(pred_dir, f"{best_step}.cif")
    if not os.path.exists(cif_src):
        cifs = sorted(glob.glob(os.path.join(pred_dir, "*.cif")), key=os.path.getmtime)
        if not cifs:
            print(f"RF3: no predicted CIF in {pred_dir}")
            metrics["RF3_Status"] = "Failed"
            return metrics, None, []
        cif_src = cifs[-1]

    cif_out = os.path.join(target_dir, f"{stem}.cif")
    shutil.copy2(cif_src, cif_out)
    pdb_out = cif_out.replace(".cif", ".pdb")
    if not cif_to_pdb(cif_out, pdb_out):
        return metrics, None, []

    # Fix protein residues in contact with the partner/ligand chain B.
    fixed = [i - 1 for i in get_residues_within_cutoff(pdb_out, "A", "B", 8.0)]
    return metrics, pdb_out, fixed


# ============================================================
# MPNN redesign through a user-installed upstream LigandMPNN checkout
# ============================================================
def run_mpnn(
    structure_file,
    fixed_residues_0idx,
    out_dir,
    num_seqs,
    temperature,
    seed,
    designed_chain="A",
):
    from rfo.cycling.mpnn.mpnn_worker import redesign

    return redesign(
        structure_file,
        fixed_residues_0idx,
        out_dir,
        num_seqs,
        temperature,
        seed,
        designed_chain,
    )


# ============================================================
# Master cycling loop
# ============================================================
def validate_dependencies(args):
    """Check the installations needed by the selected routing probability."""
    from rfo.cycling.boltz import boltz_runner
    from rfo.cycling.mpnn.mpnn_worker import settings

    settings()

    def check_process(image, executable, label):
        if image:
            if not Path(image).expanduser().is_file():
                raise FileNotFoundError(f"{label} image does not exist: {image}")
            executable = "apptainer"
        if not shutil.which(executable):
            raise FileNotFoundError(
                f"{label} executable not found: {executable}; see docs/installation.md"
            )

    if args.backprop_fraction > 0:
        check_process(RF3_SIF, RF3_PYTHON, "RF3")
        with open(args.rf3_config) as f:
            config = yaml.safe_load(f)
        loss = args.loss or config.get("loss_type", "")
        if loss not in {
            "pae_interface_mean",
            "pae_interface_min",
            "pae_mean",
            "pde_mean",
            "plddt_mean",
            "iptm",
        }:
            raise ValueError(
                "RF3 cycling requires a confidence-based loss to produce a structure for MPNN."
            )
        checkpoints = config.get("checkpoint_paths") or config.get("checkpoint_path")
        if not checkpoints:
            raise ValueError(
                "RF3 config must specify checkpoint_path or checkpoint_paths."
            )
        if isinstance(checkpoints, str):
            checkpoints = [checkpoints]
        for checkpoint in checkpoints:
            if checkpoint == "${oc.env:RF3_CKPT}":
                checkpoint = os.environ.get("RF3_CKPT", "")
                if not checkpoint:
                    raise ValueError(
                        "Set RF3_CKPT (or assets.rf3_ckpt) to a confidence-enabled RF3 checkpoint."
                    )
            if "${" not in checkpoint and not Path(checkpoint).expanduser().is_file():
                raise FileNotFoundError(f"RF3 checkpoint does not exist: {checkpoint}")
    if args.backprop_fraction < 1:
        check_process(boltz_runner.BOLTZ_SIF, boltz_runner.BOLTZ_EXECUTABLE, "Boltz")
        if (
            boltz_runner.BOLTZ_CKPT
            and not Path(boltz_runner.BOLTZ_CKPT).expanduser().is_file()
        ):
            raise FileNotFoundError(
                f"Boltz checkpoint does not exist: {boltz_runner.BOLTZ_CKPT}"
            )


def run_loop(args) -> None:
    validate_dependencies(args)
    random.seed(args.seed)
    input_dir = os.path.join(args.out_dir, "inputs")
    output_dir = args.out_dir
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(args.input_structure).stem
    current = os.path.join(input_dir, f"{stem}_0.pdb")
    if args.input_structure.endswith(".cif"):
        if not cif_to_pdb(args.input_structure, current):
            raise RuntimeError("Could not convert the input complex to PDB.")
    else:
        force_copy(args.input_structure, current)

    cyclic_chains = ["A"] if args.cyclic else None
    records: List[Dict] = []
    records_path = Path(output_dir) / f"{stem}_records.json"
    records_path.write_text("[]\n")
    cycle, attempts = 0, 0
    max_attempts = args.total_cycles * 3

    while cycle < args.total_cycles and attempts < max_attempts:
        attempts += 1
        use_rf3 = random.random() < args.backprop_fraction
        print(
            f"\n=== cycle {cycle}/{args.total_cycles} "
            f"(attempt {attempts}/{max_attempts}) "
            f"method={'RF3' if use_rf3 else 'Boltz'} ==="
        )
        t0 = time.time()

        if use_rf3:
            metrics, struct_path, fixed = run_rf3_branch(
                args.rf3_config,
                current,
                cycle,
                output_dir,
                args.cyclic,
                loss=args.loss,
                msa_paths=args.msa_paths,
                templates=args.templates,
            )
        else:
            metrics, struct_path, fixed = run_boltz_branch(
                args.af3_json_template,
                current,
                cycle,
                output_dir,
                args.num_samples,
                args.template_plddt_threshold,
                cyclic_chains,
                msa_paths=args.msa_paths,
            )

        if struct_path is None or not os.path.exists(struct_path):
            print(f"structure step failed at cycle {cycle}; retrying")
            continue

        # MPNN redesign of the binder chain.
        print("--- MPNN redesign ---")
        mpnn_dir = os.path.join(output_dir, f"recycle_{cycle + 1}", "mpnn")
        seq, redesigned = run_mpnn(
            struct_path,
            fixed,
            mpnn_dir,
            args.mpnn_num_seqs,
            args.mpnn_temperature,
            args.seed + cycle,
        )
        if redesigned is None or not os.path.exists(redesigned):
            print(f"MPNN failed at cycle {cycle}; retrying")
            continue

        # Hand-off to next cycle.
        next_input = os.path.join(input_dir, f"{stem}_{cycle + 1}.pdb")
        if redesigned.endswith(".cif"):
            if not cif_to_pdb(redesigned, next_input):
                raise RuntimeError("Could not convert the MPNN design to PDB.")
        else:
            force_copy(redesigned, next_input)
        current = next_input

        metrics.update(
            {
                "cycle": cycle,
                "mpnn_sequence": seq,
                "wall_time_sec": round(time.time() - t0, 2),
            }
        )
        records.append(metrics)
        records_path.write_text(json.dumps(records, indent=2) + "\n")
        cycle += 1

    if cycle < args.total_cycles:
        print(
            f"only {cycle}/{args.total_cycles} cycles completed "
            f"after {attempts} attempts"
        )

    with open(os.path.join(output_dir, f"{stem}_records.json"), "w") as f:
        json.dump(records, f, indent=2)
    if cycle < args.total_cycles:
        raise RuntimeError(
            f"Only {cycle}/{args.total_cycles} cycles completed; see saved records."
        )
    print(f"\nDone. Records → {os.path.join(output_dir, f'{stem}_records.json')}")
