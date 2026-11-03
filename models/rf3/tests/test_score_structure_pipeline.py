"""Pipeline-level CPU tests for structure-scoring mode.

These run the real RF3 transform pipeline (``is_inference=True``) on the bundled
test structures. No checkpoint, no GPU, no MSA directories — a few seconds per
featurization.

What is pinned here:

- The ``is_present_in_input`` annotation survives the whole pipeline (atomization,
  hydrogen/terminal-oxygen removal, feature aggregation) and still lines up
  index-for-index with ``ground_truth["coord_atom_lvl"]``.
- It records what the occupancy mask cannot: ``apply_conformer_and_template_selections``
  rewrites every NaN coordinate to ``-1``, and the scoring coordinates must never carry
  that sentinel.
- With scoring off, featurization is bit-identical to the normal prediction path and
  gains no annotation (the opt-in has no side effects).
"""

import numpy as np
import pytest
import torch
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from rf3.data.pipelines import build_af3_transform_pipeline
from rf3.data.score_structure import (
    PRESENCE_ANNOTATION,
    ScoringSpec,
    build_score_coordinates,
    summarize_scoring,
)
from rf3.utils.inference import InferenceInput

from foundry.testing import get_test_data_dir

# Resolved here rather than imported from `conftest`: the integration suite has a
# conftest of its own, and under pytest's prepend import mode a bare
# `from conftest import ...` can bind to that one instead.
TEST_DATA_DIR = get_test_data_dir(__file__)

# Mirrors the inference engine's transform overrides, minus everything that needs
# cluster data (MSA directories, the residue embedding cache).
_PIPELINE_KWARGS = dict(
    is_inference=True,
    protein_msa_dirs=[],
    rna_msa_dirs=[],
    n_recycles=1,
    crop_size=None,
    max_atoms_in_crop=None,
    n_msa=16,
    diffusion_batch_size=1,
    return_atom_array=True,
    run_confidence_head=True,
    p_unconditional=1.0,
    p_dropout_atom_level_embeddings=0.0,
    residue_cache_dir=None,
    undesired_res_names=[],
    use_element_for_atom_names_of_atomized_tokens=True,
    fallback_conformer_to_input_coords=True,
    allowed_chain_types_for_conditioning=None,
    p_give_polymer_ref_conf=0.0,
    p_give_non_polymer_ref_conf=0.0,
    p_dropout_ref_conf=0.0,
)


@pytest.fixture(scope="module")
def pipeline():
    return build_af3_transform_pipeline(**_PIPELINE_KWARGS)


def _featurize(pipeline, *, score_structure: ScoringSpec | None, seed: int = 1) -> dict:
    """Featurize `1cyo.cif` under a fixed RNG state."""
    with rng_state(create_rng_state_from_seeds(seed, seed, seed)):
        inference_input = InferenceInput.from_cif_path(
            TEST_DATA_DIR / "1cyo.cif", score_structure=score_structure
        )
        return pipeline(inference_input.to_pipeline_input())


@pytest.fixture(scope="module")
def scored_example(pipeline):
    return _featurize(pipeline, score_structure=ScoringSpec(enabled=True))


# --- the annotation survives the pipeline -----------------------------------


def test_presence_annotation_survives_the_pipeline(scored_example):
    atom_array = scored_example["atom_array"]
    assert PRESENCE_ANNOTATION in set(atom_array.get_annotation_categories())

    presence = atom_array.get_annotation(PRESENCE_ANNOTATION)
    assert presence.dtype == bool
    # Index-for-index with the coordinates the confidence head will see.
    assert len(presence) == len(scored_example["ground_truth"]["coord_atom_lvl"])
    assert (
        0 < presence.sum() < len(presence)
    ), "1cyo.cif has both given and absent atoms"


def test_the_pipeline_writes_the_minus_one_sentinel_over_absent_atoms(scored_example):
    # This is why the annotation exists: by this point "absent" is indistinguishable
    # from "at (-1, -1, -1)" in the coordinates alone.
    coords = scored_example["ground_truth"]["coord_atom_lvl"].numpy()
    presence = scored_example["atom_array"].get_annotation(PRESENCE_ANNOTATION)

    sentinel = (coords == -1.0).all(axis=-1)
    assert sentinel.sum() > 0
    assert np.array_equal(sentinel, ~presence)


def test_scoring_coordinates_never_carry_the_sentinel(scored_example):
    coords, mask = build_score_coordinates(scored_example)

    assert coords.shape == scored_example["ground_truth"]["coord_atom_lvl"].shape
    assert not (coords.numpy() == -1.0).all(axis=-1).any()
    assert not torch.isnan(coords).any()
    # Given atoms are passed through untouched.
    given = scored_example["ground_truth"]["coord_atom_lvl"][mask]
    assert torch.equal(coords[mask], given)


def test_summary_matches_the_featurized_example(scored_example):
    coords, mask = build_score_coordinates(scored_example)
    rep_atom_idxs = scored_example["ground_truth"]["rep_atom_idxs"]
    info = summarize_scoring(
        example_id="1cyo",
        atom_array=scored_example["atom_array"],
        score_mask_L=mask,
        rep_atom_idxs=rep_atom_idxs,
        diffusion_skipped=bool(mask.all()),
        structure_path=TEST_DATA_DIR / "1cyo.cif",
    )

    assert info["atoms"]["n_total"] == len(mask)
    assert info["atoms"]["n_given"] == int(mask.sum())
    assert info["representative_atoms"]["n_total"] == len(rep_atom_idxs)
    assert info["diffusion_skipped"] is False, "1cyo.cif is missing atoms"
    # 1cyo has a disordered loop (A89-A93) whose side chains, and so whose CB
    # representative atoms, are absent from the file.
    assert info["n_missing_representative_atoms"] > 0
    missing = {
        (atom["res_id"], atom["atom_name"])
        for atom in info["missing_representative_atoms"]
    }
    assert (89, "CB") in missing


def test_every_chain_of_the_input_appears_in_the_summary(scored_example):
    _, mask = build_score_coordinates(scored_example)
    info = summarize_scoring(
        example_id="1cyo",
        atom_array=scored_example["atom_array"],
        score_mask_L=mask,
        rep_atom_idxs=scored_example["ground_truth"]["rep_atom_idxs"],
        diffusion_skipped=False,
    )
    # 1cyo.cif is a protein chain plus a HEM ligand.
    chain_iids = set(np.unique(scored_example["atom_array"].chain_iid).tolist())
    assert set(info["chains"]) == chain_iids
    assert len(chain_iids) == 2


# --- the opt-in has no side effects -----------------------------------------


def _fingerprint(features: dict) -> dict:
    """Hashable fingerprint of a nested feature dict."""
    out = {}
    for key, value in sorted(features.items()):
        if isinstance(value, torch.Tensor):
            out[key] = (
                tuple(value.shape),
                str(value.dtype),
                float(value.double().sum()),
            )
        elif isinstance(value, dict):
            out[key] = _fingerprint(value)
        else:
            out[key] = repr(value)
    return out


def test_featurization_is_unchanged_when_scoring_is_off(pipeline):
    with_scoring = _featurize(pipeline, score_structure=ScoringSpec(enabled=True))
    without = _featurize(pipeline, score_structure=None)

    assert _fingerprint(without["feats"]) == _fingerprint(with_scoring["feats"])
    assert torch.equal(
        without["coord_atom_lvl_to_be_noised"],
        with_scoring["coord_atom_lvl_to_be_noised"],
    )
    assert torch.equal(
        without["ground_truth"]["coord_atom_lvl"],
        with_scoring["ground_truth"]["coord_atom_lvl"],
    )
    assert torch.equal(
        without["ground_truth"]["mask_atom_lvl"],
        with_scoring["ground_truth"]["mask_atom_lvl"],
    )


def test_no_annotation_is_added_when_scoring_is_off(pipeline):
    without = _featurize(pipeline, score_structure=None)
    disabled = _featurize(pipeline, score_structure=ScoringSpec(enabled=False))

    for example in (without, disabled):
        assert PRESENCE_ANNOTATION not in set(
            example["atom_array"].get_annotation_categories()
        )


def test_the_annotation_is_the_only_difference_in_the_atom_array(pipeline):
    with_scoring = _featurize(pipeline, score_structure=ScoringSpec(enabled=True))
    without = _featurize(pipeline, score_structure=None)

    extra = set(with_scoring["atom_array"].get_annotation_categories()) - set(
        without["atom_array"].get_annotation_categories()
    )
    assert extra == {PRESENCE_ANNOTATION}
    assert np.array_equal(with_scoring["atom_array"].coord, without["atom_array"].coord)


# --- non-polymer entities (ligands, ions, non-canonical residues) -----------
#
# `1cyo.cif` carries a HEM ligand, which the pipeline atomizes: every heavy atom
# becomes its own token, and therefore its own token representative atom. Scoring
# accounting for non-polymer entities is strictly per-atom, with none of the
# "a missing side chain cannot change the score" slack that polymers get.


def _hem_atom_indices(atom_array) -> np.ndarray:
    return np.where(np.asarray(atom_array.res_name) == "HEM")[0]


def test_every_ligand_atom_is_its_own_representative_atom(scored_example):
    atom_array = scored_example["atom_array"]
    hem_idxs = _hem_atom_indices(atom_array)
    assert hem_idxs.size > 0, "1cyo.cif should contain a HEM ligand"

    rep_idxs = scored_example["ground_truth"]["rep_atom_idxs"].numpy()
    # Every HEM atom is a representative atom ...
    assert set(hem_idxs.tolist()) <= set(rep_idxs.tolist())
    # ... and contributes its own token, unlike an amino acid's 1-per-residue.
    assert np.isin(rep_idxs, hem_idxs).sum() == hem_idxs.size


def test_a_fully_resolved_ligand_is_reported_as_complete(scored_example):
    """The HEM of `1cyo.cif` is fully resolved, so its chain must show 100% coverage."""
    coords, mask = build_score_coordinates(scored_example)
    info = summarize_scoring(
        example_id="1cyo",
        atom_array=scored_example["atom_array"],
        score_mask_L=mask,
        rep_atom_idxs=scored_example["ground_truth"]["rep_atom_idxs"],
        diffusion_skipped=False,
    )
    hem_chain = str(
        np.asarray(scored_example["atom_array"].chain_iid)[
            _hem_atom_indices(scored_example["atom_array"])
        ][0]
    )
    chain = info["chains"][hem_chain]
    assert chain["atoms"]["coverage"] == 1.0
    assert chain["representative_atoms"]["coverage"] == 1.0
    # Atom count == representative-atom count: one token per atom.
    assert chain["atoms"]["n_total"] == chain["representative_atoms"]["n_total"]
    assert chain["n_residues"] == 1


def test_a_partially_resolved_ligand_has_its_missing_atoms_named(pipeline, tmp_path):
    """Deleting ligand atoms must be reported, not papered over by the CCD template.

    `add_missing_atoms` re-adds them from the CCD with NaN coordinates, which the
    presence annotation records as absent. Were they instead filled from the reference
    conformer, scoring would silently pin the ligand to ideal geometry.
    """
    dropped = {"CMA", "CMB", "CMC", "CMD", "CHA", "CHB"}
    source = (TEST_DATA_DIR / "1cyo.cif").read_text().splitlines(keepends=True)
    kept = [
        line
        for line in source
        if not (
            line.startswith("HETATM")
            and len(line.split()) > 6
            and line.split()[5] == "HEM"
            and line.split()[3] in dropped
        )
    ]
    stripped = tmp_path / "1cyo_hem_stripped.cif"
    stripped.write_text("".join(kept))

    with rng_state(create_rng_state_from_seeds(1, 1, 1)):
        example = pipeline(
            InferenceInput.from_cif_path(
                stripped, score_structure=ScoringSpec(enabled=True)
            ).to_pipeline_input()
        )

    _, mask = build_score_coordinates(example)
    info = summarize_scoring(
        example_id="1cyo_hem_stripped",
        atom_array=example["atom_array"],
        score_mask_L=mask,
        rep_atom_idxs=example["ground_truth"]["rep_atom_idxs"],
        diffusion_skipped=False,
    )
    hem_chain = str(
        np.asarray(example["atom_array"].chain_iid)[
            _hem_atom_indices(example["atom_array"])
        ][0]
    )
    chain = info["chains"][hem_chain]
    # Each deleted atom costs one token, since every ligand atom is representative.
    assert chain["atoms"]["n_given"] == chain["atoms"]["n_total"] - len(dropped)
    assert chain["representative_atoms"]["n_given"] == (
        chain["representative_atoms"]["n_total"] - len(dropped)
    )
    assert chain["n_residues_complete"] == 0

    named = {
        entry["atom_name"]
        for entry in info["missing_representative_atoms"]
        if entry["chain"] == hem_chain
    }
    assert named == dropped


# --- a sequence/CCD input has nothing to score ------------------------------


def test_a_cif_input_reports_that_it_carries_coordinates():
    assert (
        InferenceInput.from_cif_path(
            TEST_DATA_DIR / "1cyo.cif", score_structure=ScoringSpec(enabled=True)
        ).has_input_coordinates
        is True
    )


def test_a_sequence_and_ccd_input_reports_that_it_carries_no_coordinates():
    """Guards the engine's refusal to score a JSON input.

    A CCD component is built with reference-conformer coordinates, which are *not*
    NaN, so the presence annotation marks them as given. They are ideal geometry, not
    an observed structure: scoring them would report a confidence for a ligand pose
    the user never supplied, and would pin that pose through the overlay. The engine
    therefore refuses an input whose coordinates did not come from the user, and this
    flag is the signal it uses.
    """
    inference_input = InferenceInput.from_json_dict(
        {
            "name": "ligand_only",
            "components": [
                {"seq": "SKAVKYYTLEEIQKHNNSK", "chain_id": "A"},
                {"ccd_code": "HEM"},
            ],
        },
        score_structure=ScoringSpec(enabled=True),
    )
    assert inference_input.has_input_coordinates is False
    # The reference conformer really is non-NaN, which is why the flag is needed.
    hem = inference_input.atom_array[
        np.asarray(inference_input.atom_array.res_name) == "HEM"
    ]
    assert hem.array_length() > 0
    assert not np.isnan(hem.coord).all()
