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
