"""End-to-end CLI tests for structure-scoring mode (``score_structure=true``).

Excluded from the default ``pytest`` run; see this directory's ``conftest.py`` for the
checkpoint requirement and how to run them::

    pytest models/rf3/tests/integration/test_score_structure_cli.py -m integration

What they pin:

- ``score_structure=true`` writes the standard outputs plus
  ``<id>_scoring_info.json``, and records coverage in ``summary_confidences``.
- **Self-consistency**: predict a structure, then score that prediction. The scored
  output must reproduce the prediction's coordinates exactly, the diffusion rollout
  must be skipped (a prediction provides every model atom), and the confidences must
  come back close to the prediction's own.
- **Rigid-motion invariance**: scoring a rotated and translated copy of the same
  structure gives the same numbers. RF3's confidence head reads the coordinates only
  through a representative-atom distance matrix (``use_Cb_distances: false`` in the
  released checkpoint), so this holds exactly rather than approximately.
- An incomplete input (``1cyo.cif`` is missing its disordered loop A89-A93 entirely)
  takes the overlay path: diffusion runs, the given atoms still come back exactly, and
  the filled atoms land at physical bond lengths rather than at the origin.
- Moving one chain 30 Angstrom away raises the inter-chain PAE.
- ``dump_trajectories`` and a coverage threshold the input cannot meet are refused, and
  the normal prediction path writes no scoring files.

Named ``test_score_structure_cli`` rather than ``test_score_structure`` to avoid a
pytest basename clash with the CPU unit tests under the suite's prepend import mode.

Note on determinism: the data pipeline draws random reference conformers, so two
examples folded in the *same* process do not see an identical trunk. The invariance
and self-consistency comparisons therefore each use their own ``rf3 fold`` call, every
one seeded identically by ``SPEED_FLAGS``.
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

# 1cyo.cif: cytochrome b5 (chain A) plus a HEM ligand (chain B). Residues A89-A93 are a
# disordered loop that is absent from the file altogether, so scoring the file exercises
# the partial-coverage overlay path; scoring an RF3 *prediction* of it exercises the
# diffusion-skip path, since a prediction provides every model atom.
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
    """Same mapping for the first model of an output CIF."""
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
    return json.loads(
        (out_dir / example_id / f"{example_id}_scoring_info.json").read_text()
    )


def _max_deviation(coords_a, coords_b):
    """Max per-atom distance over the atoms the two mappings share."""
    shared = sorted(set(coords_a) & set(coords_b))
    assert len(shared) > 500, f"only {len(shared)} shared atoms"
    return (
        max(float(np.linalg.norm(coords_a[key] - coords_b[key])) for key in shared),
        len(shared),
    )


def _mean_inter_chain_pae(summary):
    values = [v for row in summary["chain_pair_pae"] for v in row if v is not None]
    return float(np.mean(values))


def _run_scoring_fold(out_dir, inputs, extra_flags=()):
    out_dir, _ = run_rf3_fold(
        inputs=inputs,
        out_dir=out_dir,
        extra_flags=["score_structure=true", "diffusion_batch_size=1", *extra_flags],
    )
    return out_dir


# --- fixtures: the incomplete input (overlay path) --------------------------


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
def scored_file_dir(require_ckpt, tmp_path_factory, displaced_input):
    """Score `1cyo.cif` as-is, and its chain-displaced variant, in one call."""
    return _run_scoring_fold(
        tmp_path_factory.mktemp("rf3_scored_file"),
        [DATA_DIR / f"{EXAMPLE}.cif", displaced_input],
    )


# --- fixtures: a complete input (diffusion-skip path) -----------------------


@pytest.fixture(scope="session")
def prediction_cif(basic_folds_dir, tmp_path_factory):
    """An RF3 prediction of `1cyo`, which provides every model atom.

    Copied to `<EXAMPLE>.cif`, since `rf3 fold` derives the example id from the input
    filename and every comparison below looks the outputs up under that id.
    """
    source = basic_folds_dir / EXAMPLE / f"{EXAMPLE}_model.cif"
    assert source.exists(), source
    destination = tmp_path_factory.mktemp("rf3_prediction") / f"{EXAMPLE}.cif"
    destination.write_bytes(source.read_bytes())
    return destination


@pytest.fixture(scope="session")
def scored_prediction_dir(require_ckpt, tmp_path_factory, prediction_cif):
    """Score the prediction. Its own `rf3 fold` call, for a reproducible trunk."""
    return _run_scoring_fold(
        tmp_path_factory.mktemp("rf3_scored_pred"), [prediction_cif]
    )


@pytest.fixture(scope="session")
def scored_rotated_prediction_dir(require_ckpt, tmp_path_factory, prediction_cif):
    """Score a rigidly moved copy of the prediction, in its own `rf3 fold` call."""
    rotated = _transform_cif(
        prediction_cif,
        tmp_path_factory.mktemp("rf3_rotated_pred") / f"{EXAMPLE}.cif",
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [13.0, -7.0, 21.0],
    )
    return _run_scoring_fold(tmp_path_factory.mktemp("rf3_scored_rot"), [rotated])


# --- outputs and provenance -------------------------------------------------


def test_scoring_writes_the_standard_outputs_plus_scoring_info(scored_file_dir):
    assert_standard_outputs(scored_file_dir, EXAMPLE)
    assert (scored_file_dir / EXAMPLE / f"{EXAMPLE}_scoring_info.json").exists()


def test_scoring_info_reports_coverage_and_the_missing_representative_atoms(
    scored_file_dir,
):
    info = _scoring_info(scored_file_dir, EXAMPLE)

    assert info["example_id"] == EXAMPLE
    assert info["structure_path"].endswith(f"{EXAMPLE}.cif")
    assert 0.0 < info["atoms"]["coverage"] < 1.0, "1cyo.cif is missing atoms"
    # The disordered loop A89-A93 is absent from the file, so its CB representative
    # atoms are too.
    assert info["n_missing_representative_atoms"] > 0
    assert (89, "CB") in {
        (atom["res_id"], atom["atom_name"])
        for atom in info["missing_representative_atoms"]
    }
    assert info["diffusion_skipped"] is False, "gaps must be filled by diffusion"
    assert set(info["chains"]) == {"A_1", "B_1"}


def test_summary_confidences_records_the_scoring_provenance(scored_file_dir):
    summary = load_summary(scored_file_dir, EXAMPLE)
    scoring = summary["scoring"]

    assert scoring["enabled"] is True
    assert scoring["diffusion_skipped"] is False
    assert 0.0 < scoring["coverage"] < 1.0
    assert 0.0 < scoring["representative_atom_coverage"] < 1.0
    # The usual metrics are still reported.
    assert summary["ptm"] is not None
    assert 0.0 <= summary["overall_plddt"] <= 1.0, "RF3 reports pLDDT on 0-1"


# --- self-consistency: score RF3's own prediction ---------------------------


def test_a_complete_input_skips_the_diffusion_rollout(scored_prediction_dir):
    info = _scoring_info(scored_prediction_dir, EXAMPLE)

    assert info["atoms"]["coverage"] == 1.0
    assert info["representative_atoms"]["coverage"] == 1.0
    assert info["n_missing_representative_atoms"] == 0
    assert info["diffusion_skipped"] is True
    assert (
        load_summary(scored_prediction_dir, EXAMPLE)["scoring"]["diffusion_skipped"]
        is True
    )


def test_scoring_a_prediction_returns_that_prediction_unchanged(
    scored_prediction_dir, prediction_cif
):
    # The single most informative check: a wrong atom ordering, a dropped annotation, a
    # frame error or a dtype truncation would all move these atoms.
    deviation, n_shared = _max_deviation(
        _input_coords_by_atom(prediction_cif),
        _output_coords_by_atom(scored_prediction_dir, EXAMPLE),
    )
    # 1e-3 A is the CIF's own coordinate precision.
    assert deviation < 2e-3, f"max deviation {deviation:.4f} A over {n_shared} atoms"


def test_scoring_a_prediction_reproduces_its_own_confidences(
    scored_prediction_dir, basic_folds_dir
):
    predicted = load_summary(basic_folds_dir, EXAMPLE)
    scored = load_summary(scored_prediction_dir, EXAMPLE)

    # Not bit-identical: re-featurizing draws fresh reference conformers, which the
    # trunk sees. The gap is the noise floor for cross-tool comparisons.
    assert scored["overall_plddt"] == pytest.approx(
        predicted["overall_plddt"], abs=0.05
    )
    assert scored["ptm"] == pytest.approx(predicted["ptm"], abs=0.05)
    assert scored["iptm"] == pytest.approx(predicted["iptm"], abs=0.05)


# --- rigid-motion invariance ------------------------------------------------


def test_scoring_is_invariant_under_a_rigid_motion_of_the_input(
    scored_prediction_dir, scored_rotated_prediction_dir
):
    original = load_summary(scored_prediction_dir, EXAMPLE)
    rotated = load_summary(scored_rotated_prediction_dir, EXAMPLE)

    # The confidence head reads the coordinates only through a distance matrix, and with
    # full coverage no diffusion runs, so these must agree to float noise. The tolerance
    # covers the 1e-3 A rounding of the rewritten CIF.
    assert rotated["ptm"] == pytest.approx(original["ptm"], abs=1e-3)
    assert rotated["iptm"] == pytest.approx(original["iptm"], abs=1e-3)
    assert rotated["overall_plddt"] == pytest.approx(
        original["overall_plddt"], abs=1e-3
    )
    assert rotated["overall_pae"] == pytest.approx(original["overall_pae"], abs=1e-2)


def test_the_rotated_output_is_written_in_the_input_frame(
    scored_rotated_prediction_dir, prediction_cif
):
    # The scored structure comes back in the frame of the file it was given, not
    # re-centred onto the original.
    rotated_in = _input_coords_by_atom(prediction_cif)
    rotated_out = _output_coords_by_atom(scored_rotated_prediction_dir, EXAMPLE)
    shared = sorted(set(rotated_in) & set(rotated_out))
    displacement = np.mean(
        [np.linalg.norm(rotated_in[k] - rotated_out[k]) for k in shared]
    )
    assert displacement > 1.0, "expected the output to keep the rotated input's frame"


# --- the overlay path -------------------------------------------------------


def test_the_given_atoms_survive_the_overlay_exactly(scored_file_dir):
    # Partial coverage: diffusion runs, but every atom the file provided must still come
    # back untouched (this is what the bfloat16 rollout would otherwise quantize).
    deviation, n_shared = _max_deviation(
        _input_coords_by_atom(DATA_DIR / f"{EXAMPLE}.cif"),
        _output_coords_by_atom(scored_file_dir, EXAMPLE),
    )
    assert deviation < 2e-3, f"max deviation {deviation:.4f} A over {n_shared} atoms"


def test_the_filled_atoms_have_physical_backbone_bond_lengths(scored_file_dir):
    # The overlay fills atoms the input lacks from a diffusion sample; check the result
    # is a real structure, not atoms parked at the origin.
    produced = _output_coords_by_atom(scored_file_dir, EXAMPLE)
    lengths = []
    for res_id in range(89, 94):  # the loop absent from the file
        atoms = {name: produced.get(("A", res_id, name)) for name in ("N", "CA", "C")}
        assert all(xyz is not None for xyz in atoms.values()), res_id
        lengths.append(float(np.linalg.norm(atoms["N"] - atoms["CA"])))
        lengths.append(float(np.linalg.norm(atoms["CA"] - atoms["C"])))

    assert max(lengths) < 2.0, lengths
    assert 1.3 < float(np.median(lengths)) < 1.7, lengths


def test_displacing_a_chain_raises_the_inter_chain_pae(scored_file_dir):
    original = load_summary(scored_file_dir, EXAMPLE)
    displaced = load_summary(scored_file_dir, f"{EXAMPLE}_displaced")

    assert _mean_inter_chain_pae(displaced) > _mean_inter_chain_pae(
        original
    ), "moving the ligand 30 A off the protein must raise the inter-chain PAE"


# --- guard rails and the normal path ----------------------------------------


def _fold_expecting_failure(tmp_path, extra_flags):
    return subprocess.run(
        [str(_RF3_BIN), "fold"]
        + SPEED_FLAGS
        + [
            f"ckpt_path={CKPT_PATH}",
            f"inputs={DATA_DIR / f'{EXAMPLE}.cif'}",
            f"out_dir={tmp_path}",
            "score_structure=true",
            *extra_flags,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )


def test_scoring_refuses_to_dump_trajectories(require_ckpt, tmp_path):
    result = _fold_expecting_failure(tmp_path, ["dump_trajectories=true"])
    assert result.returncode != 0
    assert "dump_trajectories" in result.stdout + result.stderr


def test_scoring_refuses_an_input_below_the_coverage_threshold(require_ckpt, tmp_path):
    result = _fold_expecting_failure(
        tmp_path, ["score_structure_min_rep_atom_coverage=1.0"]
    )
    assert result.returncode != 0
    assert "representative atom" in result.stdout + result.stderr


def test_scoring_refuses_a_sequence_and_ccd_input(require_ckpt, tmp_path):
    """A JSON input has no coordinates to score.

    Its CCD components do carry reference-conformer coordinates, which are ideal
    geometry rather than an observed structure; scoring them would report a confidence
    for a ligand pose nobody supplied. The engine must refuse rather than oblige.
    """
    result = subprocess.run(
        [str(_RF3_BIN), "fold"]
        + SPEED_FLAGS
        + [
            f"ckpt_path={CKPT_PATH}",
            f"inputs={DATA_DIR / '1cyo_with_ligand.json'}",
            f"out_dir={tmp_path}",
            "score_structure=true",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "no input coordinates" in output
    assert "reference conformer" in output


def test_the_normal_prediction_path_writes_no_scoring_files(basic_folds_dir):
    # Reuses the plain-fold fixture: scoring must be fully opt-in.
    for example_id in (EXAMPLE, f"{EXAMPLE}_from_json"):
        directory = basic_folds_dir / example_id
        assert not (directory / f"{example_id}_scoring_info.json").exists()
        assert "scoring" not in load_summary(basic_folds_dir, example_id)
