"""End-to-end CLI tests for structure-scoring mode (``score_structure=true``).

Excluded from the default ``pytest`` run; see this directory's ``conftest.py`` for the
checkpoint requirement and how to run them::

    pytest models/rf3/tests/integration/test_score_structure.py -m integration

These run on CPU. What they pin:

- ``score_structure=true`` writes the standard outputs plus
  ``<id>_scoring_info.json``, and records coverage in ``summary_confidences``.
- **The scored output structure reproduces the input coordinates.** This is the cheap
  proxy for the GPU self-consistency test: a wrong atom ordering, a dropped
  annotation or a frame error all show up here immediately.
- Scores are invariant under a rigid motion of the input. RF3's confidence head reads
  the coordinates only through a representative-atom distance matrix
  (``use_Cb_distances: false`` in the released checkpoint), so this must hold exactly,
  not just approximately.
- Moving one chain 30 Angstrom away collapses the interface confidence while leaving
  intra-chain pLDDT roughly intact.
- ``dump_trajectories`` is refused, and the normal prediction path writes no scoring
  files.
"""

import json
import subprocess

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from atomworks.io import parse
from conftest import (
    _RF3_BIN,
    CKPT_PATH,
    DATA_DIR,
    SPEED_FLAGS,
    assert_standard_outputs,
    load_summary,
    run_rf3_fold,
)

pytestmark = pytest.mark.integration

# 1cyo.cif: cytochrome b5 (chain A) plus a HEM ligand (chain B). Its residues A89-A93
# are a disordered loop whose side chains are absent from the file, so scoring it
# exercises the partial-coverage overlay path rather than the diffusion-skip path.
EXAMPLE = "1cyo"


# --- helpers ----------------------------------------------------------------


def _transform_cif(src, dst, rotation, translation, *, chain=None):
    """Copy a CIF, rigidly moving its coordinates (optionally only one chain).

    Operates on the raw ``atom_site`` category so every other category survives
    untouched and the file re-parses exactly as the original does.
    """
    cif = pdbx.CIFFile.read(str(src))
    category = cif.block["atom_site"]
    keys = ("Cartn_x", "Cartn_y", "Cartn_z")
    coords = np.stack([category[k].as_array(np.float64) for k in keys], axis=-1)

    selection = (
        np.ones(len(coords), dtype=bool)
        if chain is None
        else category["label_asym_id"].as_array(str) == chain
    )
    coords[selection] = coords[selection] @ np.asarray(rotation).T + np.asarray(
        translation
    )

    for key, column in zip(keys, coords.T, strict=True):
        category[key] = pdbx.CIFColumn(np.round(column, 3))
    cif.write(str(dst))
    return dst


def _input_coords_by_atom(path):
    """Map (chain_id, res_id, atom_name) -> coordinate for the atoms in a file."""
    parsed = parse(path, hydrogen_policy="remove", add_missing_atoms=False)
    array = (
        parsed["assemblies"]["1"][0]
        if "assemblies" in parsed
        else parsed["asym_unit"][0]
    )
    return {
        (str(c), int(r), str(a)): xyz
        for c, r, a, xyz in zip(
            array.chain_id, array.res_id, array.atom_name, array.coord, strict=True
        )
    }


def _output_coords_by_atom(out_dir, example_id):
    """Same mapping for the first model of a scored output CIF."""
    cif = pdbx.CIFFile.read(str(out_dir / example_id / f"{example_id}_model.cif"))
    category = cif.block["atom_site"]
    model = category["pdbx_PDB_model_num"].as_array(int)
    first = model == model[0]
    chains = category["label_asym_id"].as_array(str)[first]
    res_ids = category["label_seq_id"].as_array(str)[first]
    names = category["label_atom_id"].as_array(str)[first]
    coords = np.stack(
        [
            category[k].as_array(np.float64)[first]
            for k in ("Cartn_x", "Cartn_y", "Cartn_z")
        ],
        axis=-1,
    )
    out = {}
    for chain, res_id, name, xyz in zip(chains, res_ids, names, coords, strict=True):
        if res_id in (".", "?", ""):
            continue
        out[(chain, int(res_id), name)] = xyz
    return out


def _scoring_info(out_dir, example_id):
    path = out_dir / example_id / f"{example_id}_scoring_info.json"
    return json.loads(path.read_text())


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="session")
def rotated_input(tmp_path_factory):
    """`1cyo.cif` rigidly moved: a 90 degree turn about z plus a translation."""
    rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    return _transform_cif(
        DATA_DIR / f"{EXAMPLE}.cif",
        tmp_path_factory.mktemp("rf3_score_rotated") / f"{EXAMPLE}_rotated.cif",
        rotation,
        [13.0, -7.0, 21.0],
    )


@pytest.fixture(scope="session")
def displaced_input(tmp_path_factory):
    """`1cyo.cif` with the HEM ligand chain pushed 30 Angstrom off the protein."""
    return _transform_cif(
        DATA_DIR / f"{EXAMPLE}.cif",
        tmp_path_factory.mktemp("rf3_score_displaced") / f"{EXAMPLE}_displaced.cif",
        np.eye(3),
        [30.0, 0.0, 0.0],
        chain="B",
    )


@pytest.fixture(scope="session")
def scored_dir(require_ckpt, tmp_path_factory, rotated_input, displaced_input):
    """One `rf3 fold` call scoring the original, rotated and displaced inputs."""
    out_dir = tmp_path_factory.mktemp("rf3_scored")
    out_dir, _ = run_rf3_fold(
        inputs=[DATA_DIR / f"{EXAMPLE}.cif", rotated_input, displaced_input],
        out_dir=out_dir,
        extra_flags=["score_structure=true", "diffusion_batch_size=1"],
    )
    return out_dir


# --- outputs ----------------------------------------------------------------


def test_scoring_writes_the_standard_outputs_plus_scoring_info(scored_dir):
    assert_standard_outputs(scored_dir, EXAMPLE)
    assert (scored_dir / EXAMPLE / f"{EXAMPLE}_scoring_info.json").exists()


def test_scoring_info_reports_coverage_and_the_missing_representative_atoms(scored_dir):
    info = _scoring_info(scored_dir, EXAMPLE)

    assert info["example_id"] == EXAMPLE
    assert 0.0 < info["atoms"]["coverage"] < 1.0, "1cyo.cif is missing some atoms"
    # 1cyo's disordered loop A89-A93 has no side chains, hence no CB representative atom.
    assert info["n_missing_representative_atoms"] > 0
    assert (89, "CB") in {
        (atom["res_id"], atom["atom_name"])
        for atom in info["missing_representative_atoms"]
    }
    # Incomplete coverage, so diffusion ran to fill the gaps.
    assert info["diffusion_skipped"] is False
    assert set(info["chains"]) == {"A_1", "B_1"}


def test_summary_confidences_records_the_scoring_provenance(scored_dir):
    summary = load_summary(scored_dir, EXAMPLE)
    scoring = summary["scoring"]

    assert scoring["enabled"] is True
    assert scoring["diffusion_skipped"] is False
    assert 0.0 < scoring["coverage"] <= 1.0
    assert 0.0 < scoring["representative_atom_coverage"] <= 1.0
    # The usual metrics are still reported.
    assert summary["ptm"] is not None
    assert 0.0 <= summary["overall_plddt"] <= 1.0, "RF3 reports pLDDT on 0-1"


# --- the scored structure is the input structure ----------------------------


def test_the_scored_output_reproduces_the_input_coordinates(scored_dir):
    # The single most informative check: a wrong atom ordering, a dropped annotation
    # or a frame error would all move these atoms.
    given = _input_coords_by_atom(DATA_DIR / f"{EXAMPLE}.cif")
    produced = _output_coords_by_atom(scored_dir, EXAMPLE)

    shared = set(given) & set(produced)
    assert len(shared) > 500, "expected most of the input's atoms in the output"
    deviations = np.array(
        [np.linalg.norm(given[key] - produced[key]) for key in sorted(shared)]
    )
    # 1e-3 A is the CIF's own coordinate precision.
    assert deviations.max() < 2e-3, f"max deviation {deviations.max():.4f} A"


def test_the_filled_atoms_have_physical_backbone_bond_lengths(scored_dir):
    # The overlay fills atoms the input lacks from a diffusion sample; check the result
    # is a real structure, not atoms parked at the origin.
    produced = _output_coords_by_atom(scored_dir, EXAMPLE)
    lengths = []
    for res_id in range(89, 94):  # the disordered loop
        n = produced.get(("A", res_id, "N"))
        ca = produced.get(("A", res_id, "CA"))
        c = produced.get(("A", res_id, "C"))
        if n is not None and ca is not None:
            lengths.append(np.linalg.norm(n - ca))
        if ca is not None and c is not None:
            lengths.append(np.linalg.norm(ca - c))

    assert lengths, "expected the disordered loop in the output"
    assert all(1.2 < length < 1.7 for length in lengths), lengths


# --- invariance and controls -------------------------------------------------


def test_scores_are_invariant_under_a_rigid_motion_of_the_input(scored_dir):
    original = load_summary(scored_dir, EXAMPLE)
    rotated = load_summary(scored_dir, f"{EXAMPLE}_rotated")

    # Exact up to float noise: the confidence head sees the coordinates only through a
    # distance matrix. The small tolerance covers the 1e-3 A rounding in the rewritten
    # CIF and the diffusion fill of the missing atoms.
    assert rotated["ptm"] == pytest.approx(original["ptm"], abs=0.01)
    assert rotated["overall_plddt"] == pytest.approx(
        original["overall_plddt"], abs=0.01
    )
    assert rotated["overall_pae"] == pytest.approx(original["overall_pae"], abs=0.2)


def test_displacing_a_chain_collapses_the_interface_confidence(scored_dir):
    original = load_summary(scored_dir, EXAMPLE)
    displaced = load_summary(scored_dir, f"{EXAMPLE}_displaced")

    def _inter_chain_pae(summary):
        matrix = summary["chain_pair_pae"]
        values = [v for row in matrix for v in row if v is not None]
        return float(np.mean(values))

    assert _inter_chain_pae(displaced) > _inter_chain_pae(
        original
    ), "moving the ligand 30 A off the protein must raise the inter-chain PAE"
    # The protein chain itself did not move, so its own pLDDT should hold up.
    chain_a_index = 0
    assert displaced["chain_ptm"][chain_a_index] == pytest.approx(
        original["chain_ptm"][chain_a_index], abs=0.15
    )


# --- guard rails and the normal path ----------------------------------------


def test_scoring_refuses_to_dump_trajectories(require_ckpt, tmp_path):
    result = subprocess.run(
        [str(_RF3_BIN), "fold"]
        + SPEED_FLAGS
        + [
            f"ckpt_path={CKPT_PATH}",
            f"inputs={DATA_DIR / f'{EXAMPLE}.cif'}",
            f"out_dir={tmp_path}",
            "score_structure=true",
            "dump_trajectories=true",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode != 0
    assert "dump_trajectories" in result.stdout + result.stderr


def test_scoring_refuses_an_input_below_the_coverage_threshold(require_ckpt, tmp_path):
    result = subprocess.run(
        [str(_RF3_BIN), "fold"]
        + SPEED_FLAGS
        + [
            f"ckpt_path={CKPT_PATH}",
            f"inputs={DATA_DIR / f'{EXAMPLE}.cif'}",
            f"out_dir={tmp_path}",
            "score_structure=true",
            "score_structure_min_rep_atom_coverage=1.0",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode != 0
    assert "representative atom" in (result.stdout + result.stderr)


def test_the_normal_prediction_path_writes_no_scoring_files(basic_folds_dir):
    # Reuses the plain-fold fixture: scoring must be fully opt-in.
    for example_id in ("1cyo", "1cyo_from_json"):
        directory = basic_folds_dir / example_id
        assert not (directory / f"{example_id}_scoring_info.json").exists()
        summary = load_summary(basic_folds_dir, example_id)
        assert "scoring" not in summary
