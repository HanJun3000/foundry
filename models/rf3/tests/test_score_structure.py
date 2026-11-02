"""CPU unit tests for structure-scoring mode (``score_structure``).

Scoring mode evaluates the confidence head on user-supplied coordinates instead of
on RF3's own diffusion samples. These tests need no checkpoint and no GPU.

What is pinned here:

- ``overlay_score_coordinates`` superposes each diffusion sample onto the given atoms
  and then overwrites them, so the given atoms come back exactly and the atoms the
  input does not provide land at their true positions (to float precision) when the
  "sample" is a rigid motion of the truth.
- ``build_score_coordinates`` reads the ``is_present_in_input`` annotation, and never
  lets the ``-1`` sentinel that :func:`rf3.utils.inference.apply_conformer_and_template_selections`
  writes over absent atoms escape into the returned coordinates.
- ``check_scoring_coverage`` warns, and names the residues, when the input is missing
  token representative atoms — the only coordinates RF3's confidence head actually
  reads — and refuses only below an explicit coverage threshold. It does not refuse by
  default: every structure in ``tests/data`` is missing some CB atoms (disordered loops,
  N-terminal residues), so a hard default would reject ordinary crystal structures.
- ``RF3WithConfidence.forward`` skips the diffusion rollout when the input covers every
  atom, overlays when it does not, and is unchanged when scoring is off.
- With scoring off, the featurization is bit-identical to the normal prediction path
  and gains no annotation.
"""

import json
import logging
from collections import deque

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from rf3.data.score_structure import (
    PRESENCE_ANNOTATION,
    ScoringCoordinates,
    ScoringSpec,
    build_score_coordinates,
    check_scoring_coverage,
    log_scoring_summary,
    overlay_score_coordinates,
    record_input_presence,
    summarize_scoring,
)

# --- helpers ----------------------------------------------------------------


def _random_rotation(seed: int) -> torch.Tensor:
    """A proper rotation matrix (det = +1), float64."""
    generator = torch.Generator().manual_seed(seed)
    u, _, v = torch.linalg.svd(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    rotation = u @ v
    if torch.linalg.det(rotation) < 0:
        flip = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
        rotation = u @ flip @ v
    return rotation


def _rigidly_moved(coords: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply a random rotation and translation, returning the input's dtype."""
    rotation = _random_rotation(seed)
    translation = torch.randn(
        3, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64
    )
    moved = coords.to(torch.float64) @ rotation.T + 10.0 * translation
    return moved.to(coords.dtype)


def _toy_atom_array(n_residues: int = 4) -> AtomArray:
    """Two chains of glycine-like residues with N, CA, C, CB per residue."""
    atom_names = ["N", "CA", "C", "CB"]
    n_atoms = 2 * n_residues * len(atom_names)
    array = AtomArray(length=n_atoms)
    chain_ids, res_ids, names = [], [], []
    for chain in ("A", "B"):
        for res_id in range(1, n_residues + 1):
            for name in atom_names:
                chain_ids.append(chain)
                res_ids.append(res_id)
                names.append(name)
    array.set_annotation("chain_id", np.array(chain_ids))
    array.set_annotation("chain_iid", np.array([f"{c}_1" for c in chain_ids]))
    array.set_annotation("res_id", np.array(res_ids))
    array.set_annotation("res_name", np.array(["GLY"] * n_atoms))
    array.set_annotation("atom_name", np.array(names))
    array.coord = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3)
    return array


def _cb_indices(atom_array: AtomArray) -> np.ndarray:
    return np.where(atom_array.atom_name == "CB")[0]


# --- overlay_score_coordinates ---------------------------------------------


def test_overlay_with_full_mask_returns_the_given_structure_exactly():
    truth = torch.randn(40, 3, generator=torch.Generator().manual_seed(0)) * 10
    sample = _rigidly_moved(truth, seed=3).unsqueeze(0).repeat(3, 1, 1)
    mask = torch.ones(40, dtype=torch.bool)

    out = overlay_score_coordinates(sample, truth, mask)

    assert out.shape == (3, 40, 3)
    assert torch.equal(out, truth.unsqueeze(0).repeat(3, 1, 1))


def test_overlay_places_missing_atoms_at_their_true_positions():
    # The "sample" is a random rigid motion of the truth, so a correct superposition
    # must bring the atoms outside the mask back onto the truth.
    truth = torch.randn(60, 3, generator=torch.Generator().manual_seed(1)) * 10
    sample = _rigidly_moved(truth, seed=7).unsqueeze(0).repeat(2, 1, 1)
    mask = torch.ones(60, dtype=torch.bool)
    mask[10:20] = False

    out = overlay_score_coordinates(sample, truth, mask)

    # Given atoms: exact. Filled atoms: recovered to float precision.
    assert torch.equal(out[:, mask], truth[mask].unsqueeze(0).repeat(2, 1, 1))
    assert (out[:, ~mask] - truth[~mask]).abs().max() < 1e-4


def test_overlay_is_invariant_to_the_frame_of_the_sample():
    truth = torch.randn(50, 3, generator=torch.Generator().manual_seed(2)) * 10
    mask = torch.ones(50, dtype=torch.bool)
    mask[:5] = False
    base = _rigidly_moved(truth, seed=11).unsqueeze(0)

    first = overlay_score_coordinates(base, truth, mask)
    second = overlay_score_coordinates(
        _rigidly_moved(base[0], seed=23).unsqueeze(0), truth, mask
    )

    assert (first - second).abs().max() < 1e-3


def test_overlay_never_leaves_a_filled_atom_on_the_sentinel_or_the_origin():
    # Mimic the pipeline: absent atoms arrive at (-1, -1, -1).
    truth = torch.randn(30, 3, generator=torch.Generator().manual_seed(4)) * 10
    mask = torch.ones(30, dtype=torch.bool)
    mask[3:9] = False
    given = truth.clone()
    given[~mask] = -1.0
    sample = _rigidly_moved(truth, seed=5).unsqueeze(0)

    out = overlay_score_coordinates(sample, given, mask)

    filled = out[0, ~mask]
    assert not torch.isclose(filled, torch.full_like(filled, -1.0)).all(dim=-1).any()
    assert not (filled.abs() < 1e-6).all(dim=-1).any()


def test_overlay_returns_the_dtype_of_the_given_structure():
    truth = torch.randn(20, 3)
    sample = truth.unsqueeze(0).to(torch.float32)
    out = overlay_score_coordinates(sample, truth, torch.ones(20, dtype=torch.bool))
    assert out.dtype == torch.float32


def test_overlay_does_not_quantize_the_given_atoms_to_the_sample_precision():
    # The diffusion rollout runs under bfloat16 autocast. Returning the sample's dtype
    # would round the input coordinates to ~3 significant digits, moving every given
    # atom by ~0.1 A at typical radii — the scored structure would not be the input.
    truth = torch.randn(40, 3, generator=torch.Generator().manual_seed(21)) * 30
    mask = torch.ones(40, dtype=torch.bool)
    mask[:6] = False
    sample = _rigidly_moved(truth, seed=22).unsqueeze(0).to(torch.bfloat16)

    out = overlay_score_coordinates(sample, truth, mask)

    assert out.dtype == torch.float32
    assert torch.equal(out[0, mask], truth[mask])


@pytest.mark.parametrize(
    ("n_given", "match"),
    [(0, "at least 3 given atoms"), (2, "at least 3 given atoms")],
)
def test_overlay_rejects_too_few_given_atoms(n_given, match):
    truth = torch.randn(20, 3)
    sample = truth.unsqueeze(0)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[:n_given] = True
    with pytest.raises(ValueError, match=match):
        overlay_score_coordinates(sample, truth, mask)


def test_overlay_rejects_shape_mismatches():
    sample = torch.randn(2, 20, 3)
    with pytest.raises(ValueError, match=r"shape \[20, 3\]"):
        overlay_score_coordinates(
            sample, torch.randn(19, 3), torch.ones(20, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match=r"shape \[20\]"):
        overlay_score_coordinates(
            sample, torch.randn(20, 3), torch.ones(19, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match=r"\[D, L, 3\]"):
        overlay_score_coordinates(
            torch.randn(20, 3), torch.randn(20, 3), torch.ones(20, dtype=torch.bool)
        )


# --- record_input_presence / build_score_coordinates ------------------------


def test_record_input_presence_marks_nan_atoms_as_absent():
    array = _toy_atom_array()
    array.coord[5] = np.nan
    array.coord[9, 1] = np.nan  # a single NaN component is enough to be absent

    record_input_presence(array)

    presence = array.get_annotation(PRESENCE_ANNOTATION)
    assert presence.dtype == bool
    assert not presence[5] and not presence[9]
    assert presence.sum() == len(array) - 2


def _pipeline_output_with_sentinels(absent: list[int]) -> tuple[dict, torch.Tensor]:
    """Build a minimal featurized example mimicking the real pipeline's output."""
    array = _toy_atom_array()
    truth = torch.tensor(array.coord.copy())
    presence = np.ones(len(array), dtype=bool)
    presence[absent] = False
    array.set_annotation(PRESENCE_ANNOTATION, presence)
    # The pipeline writes -1 over every absent atom and zeroes their occupancy.
    array.coord[absent] = -1.0
    occupancy = np.ones(len(array), dtype=np.float32)
    occupancy[absent] = 0.0
    array.set_annotation("occupancy", occupancy)

    pipeline_output = {
        "atom_array": array,
        "ground_truth": {
            "coord_atom_lvl": torch.tensor(array.coord),
            "mask_atom_lvl": torch.tensor(occupancy > 0.0),
            "rep_atom_idxs": torch.tensor(_cb_indices(array)),
        },
    }
    return pipeline_output, truth


def test_build_score_coordinates_masks_the_sentinel_atoms():
    absent = [2, 7, 13]
    pipeline_output, truth = _pipeline_output_with_sentinels(absent)

    coords, mask = build_score_coordinates(pipeline_output)

    assert coords.shape == (len(pipeline_output["atom_array"]), 3)
    assert mask.dtype == torch.bool
    assert mask.sum() == len(mask) - len(absent)
    assert not mask[absent].any()
    # Masked-out entries are zeroed, so no consumer can read the -1 sentinel.
    assert torch.equal(coords[~mask], torch.zeros(len(absent), 3))
    # Everything else is the input structure, untouched.
    assert torch.equal(coords[mask], truth[mask])


def test_build_score_coordinates_falls_back_to_the_occupancy_mask(caplog):
    pipeline_output, _ = _pipeline_output_with_sentinels([1, 4])
    pipeline_output["atom_array"].del_annotation(PRESENCE_ANNOTATION)

    with caplog.at_level(logging.WARNING):
        _, mask = build_score_coordinates(pipeline_output)

    assert mask.sum() == len(mask) - 2
    assert PRESENCE_ANNOTATION in caplog.text


def test_build_score_coordinates_rejects_an_example_with_no_given_atoms():
    pipeline_output, _ = _pipeline_output_with_sentinels(list(range(32)))
    with pytest.raises(ValueError, match="No input coordinates to score"):
        build_score_coordinates(pipeline_output)


def test_build_score_coordinates_requires_ground_truth_coordinates():
    with pytest.raises(KeyError, match="coord_atom_lvl"):
        build_score_coordinates({"ground_truth": {}})


# --- summarize_scoring / check_scoring_coverage -----------------------------


def _summary(absent: list[int], diffusion_skipped: bool = False) -> dict:
    pipeline_output, _ = _pipeline_output_with_sentinels(absent)
    _, mask = build_score_coordinates(pipeline_output)
    return summarize_scoring(
        example_id="toy",
        atom_array=pipeline_output["atom_array"],
        score_mask_L=mask,
        rep_atom_idxs=pipeline_output["ground_truth"]["rep_atom_idxs"],
        diffusion_skipped=diffusion_skipped,
        structure_path="/tmp/toy.cif",
    )


def test_summary_reports_complete_coverage_for_a_complete_structure():
    info = _summary([], diffusion_skipped=True)

    assert info["atoms"]["coverage"] == 1.0
    assert info["representative_atoms"]["coverage"] == 1.0
    assert info["diffusion_skipped"] is True
    assert set(info["chains"]) == {"A_1", "B_1"}
    for chain in info["chains"].values():
        assert chain["n_residues"] == 4
        assert chain["n_residues_mapped"] == 4
        assert chain["n_residues_complete"] == 4
    assert json.loads(json.dumps(info))  # JSON-serializable


def test_summary_separates_atom_coverage_from_representative_atom_coverage():
    array = _toy_atom_array()
    # Drop the CB of chain A residue 1 (a representative atom) and the N of chain A
    # residue 2 (not one), so two distinct residues become incomplete.
    in_chain_a = array.chain_id == "A"
    cb_of_res_1 = int(
        np.where(in_chain_a & (array.res_id == 1) & (array.atom_name == "CB"))[0][0]
    )
    n_of_res_2 = int(
        np.where(in_chain_a & (array.res_id == 2) & (array.atom_name == "N"))[0][0]
    )
    info = _summary([cb_of_res_1, n_of_res_2])

    assert info["atoms"]["n_given"] == len(array) - 2
    # Only the CB counts against representative-atom coverage.
    assert info["representative_atoms"]["n_given"] == 7
    assert info["representative_atoms"]["n_total"] == 8
    chain_a = info["chains"]["A_1"]
    assert chain_a["n_residues_mapped"] == 4
    assert chain_a["n_residues_complete"] == 2
    # Chain B is untouched.
    assert info["chains"]["B_1"]["atoms"]["coverage"] == 1.0


def test_summary_names_the_residues_missing_a_representative_atom():
    array = _toy_atom_array()
    in_chain_b = array.chain_id == "B"
    cb_of_res_3 = int(
        np.where(in_chain_b & (array.res_id == 3) & (array.atom_name == "CB"))[0][0]
    )
    info = _summary([cb_of_res_3])

    assert info["n_missing_representative_atoms"] == 1
    assert info["missing_representative_atoms"] == [
        {"chain": "B_1", "res_id": 3, "res_name": "GLY", "atom_name": "CB"}
    ]


def test_check_coverage_is_silent_when_every_representative_atom_is_given(caplog):
    array = _toy_atom_array()
    n_index = int(np.where(array.atom_name == "N")[0][0])
    with caplog.at_level(logging.WARNING):
        check_scoring_coverage(_summary([n_index]))
    assert caplog.text == ""


def test_check_coverage_only_warns_by_default(caplog):
    # Default must not refuse: real structures routinely lack a few CB atoms.
    info = _summary([int(_cb_indices(_toy_atom_array())[0])])
    with caplog.at_level(logging.WARNING):
        check_scoring_coverage(info)
    assert "representative atoms" in caplog.text
    assert "GLY1/CB" in caplog.text  # names the affected residue


def test_check_coverage_refuses_below_an_explicit_threshold():
    info = _summary([int(_cb_indices(_toy_atom_array())[0])])
    assert info["representative_atoms"]["coverage"] == pytest.approx(7 / 8)
    with pytest.raises(ValueError, match="below the required"):
        check_scoring_coverage(info, min_representative_atom_coverage=1.0)
    # Just inside the threshold: a warning, not a refusal.
    check_scoring_coverage(info, min_representative_atom_coverage=0.5)


def test_log_scoring_summary_writes_one_line_per_chain_plus_a_total(caplog):
    with caplog.at_level(logging.INFO):
        log_scoring_summary(_summary([], diffusion_skipped=True))
    lines = [r.message for r in caplog.records]
    assert sum("chain A_1 -> model chain A_1" in line for line in lines) == 1
    assert sum("chain B_1 -> model chain B_1" in line for line in lines) == 1
    assert any("diffusion skipped" in line for line in lines)


# --- ScoringSpec ------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_scoring_spec_from_bool(value):
    spec = ScoringSpec.from_json_value(value)
    assert spec.enabled is value
    assert spec.min_representative_atom_coverage == 0.0


def test_scoring_spec_from_mapping_defaults_to_enabled():
    spec = ScoringSpec.from_json_value({"min_representative_atom_coverage": 0.9})
    assert spec.enabled is True
    assert spec.min_representative_atom_coverage == 0.9


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_scoring_spec_rejects_a_coverage_threshold_outside_zero_to_one(value):
    with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
        ScoringSpec(enabled=True, min_representative_atom_coverage=value)


def test_scoring_spec_can_be_disabled_explicitly_in_a_mapping():
    assert ScoringSpec.from_json_value({"enabled": False}).enabled is False


def test_scoring_spec_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        ScoringSpec.from_json_value({"structurepath": "x.cif"}, example_id="job1")


def test_scoring_spec_rejects_a_non_mapping():
    with pytest.raises(ValueError, match="boolean or a mapping"):
        ScoringSpec.from_json_value("yes")


@pytest.mark.parametrize(
    "value",
    [{"structure_path": "x.cif"}, {"structurePath": "x.cif"}, {"chainIds": ["A"]}],
)
def test_scoring_spec_rejects_the_not_yet_supported_external_file_form(value):
    with pytest.raises(NotImplementedError, match="not yet"):
        ScoringSpec.from_json_value(value)


# --- the model hook ---------------------------------------------------------


class _RecordingConfidenceHead:
    """Stands in for ConfidenceHead, recording the coordinates it is handed."""

    def __init__(self, n_token: int):
        self.n_token = n_token
        self.seen: list[torch.Tensor] = []

    def __call__(
        self, s_inputs, s_trunk, z_trunk, x_pred, seq, rep_atoms, frame_atom_idxs=None
    ):
        self.seen.append(None if x_pred is None else x_pred.clone())
        return {
            "plddt_logits": torch.zeros(1, self.n_token, 2),
            "pae_logits": torch.zeros(1, self.n_token, self.n_token, 2),
            "pde_logits": torch.zeros(1, self.n_token, self.n_token, 2),
            "exp_resolved_logits": torch.zeros(1, self.n_token, 2),
        }


class _RecordingSampler:
    """Stands in for the inference sampler, recording whether it ran."""

    def __init__(self, coords: torch.Tensor):
        self.coords = coords
        self.n_calls = 0

    def sample_diffusion_like_af3(self, **kwargs):
        self.n_calls += 1
        return {
            "X_L": self.coords.clone(),
            "X_noisy_L_traj": [self.coords.clone()],
            "X_denoised_L_traj": [self.coords.clone()],
            "t_hats": [torch.zeros(self.coords.shape[0])],
        }


def _forward_with_stubs(
    *,
    n_atom=24,
    n_token=8,
    n_sample=2,
    score_coordinates=None,
    sample_coords=None,
):
    """Call the real ``RF3WithConfidence.forward`` with every heavy part stubbed out."""
    from rf3.model.RF3 import RF3WithConfidence

    model = object.__new__(RF3WithConfidence)
    model.training = False
    if sample_coords is None:
        sample_coords = torch.randn(
            n_sample, n_atom, 3, generator=torch.Generator().manual_seed(9)
        )
    sampler = _RecordingSampler(sample_coords)
    head = _RecordingConfidenceHead(n_token)
    model.inference_sampler = sampler
    model.mini_rollout_sampler = _RecordingSampler(sample_coords)
    model.diffusion_module = object()  # only forwarded to the (stubbed) sampler
    model.confidence_head = head
    model.distogram_head = lambda z: torch.zeros(n_token, n_token, 2)
    model.trunk_forward_with_recycling = lambda f, n_recycles: iter(
        [
            {
                "S_inputs_I": torch.zeros(n_token, 4),
                "S_init_I": torch.zeros(n_token, 4),
                "Z_init_II": torch.zeros(n_token, n_token, 4),
                "S_I": torch.zeros(n_token, 4),
                "Z_II": torch.zeros(n_token, n_token, 4),
            }
        ]
    )

    network_input = {
        "f": {},
        "t": torch.zeros(n_sample),
        "seq": torch.zeros(n_token, dtype=torch.long),
        "rep_atom_idxs": torch.arange(n_token),
        "frame_atom_idxs": torch.zeros(n_token, 3, dtype=torch.long),
    }
    out = RF3WithConfidence.forward(
        model,
        input=network_input,
        n_cycle=1,
        coord_atom_lvl_to_be_noised=torch.zeros(n_sample, n_atom, 3),
        score_coordinates=score_coordinates,
    )
    return out, sampler, head


def test_forward_without_scoring_runs_diffusion_and_scores_the_sample():
    out, sampler, head = _forward_with_stubs()

    assert sampler.n_calls == 1
    assert len(head.seen) == 2
    assert torch.equal(out["X_pred_rollout_L"], sampler.coords)
    for i, seen in enumerate(head.seen):
        assert torch.equal(seen, sampler.coords[i].unsqueeze(0))


def test_forward_skips_diffusion_when_the_input_covers_every_atom():
    given = torch.randn(24, 3, generator=torch.Generator().manual_seed(12))
    out, sampler, head = _forward_with_stubs(
        score_coordinates=ScoringCoordinates(given, torch.ones(24, dtype=torch.bool))
    )

    assert sampler.n_calls == 0, "diffusion must be skipped at full coverage"
    assert torch.equal(out["X_pred_rollout_L"], given.unsqueeze(0).repeat(2, 1, 1))
    for seen in head.seen:
        assert torch.equal(seen, given.unsqueeze(0))
    # Nothing was rolled out, so there is no trajectory to report.
    assert out["X_noisy_L_traj"] == []
    assert out["t_hats"] == []


def test_forward_overlays_when_the_input_is_missing_atoms():
    given = torch.randn(24, 3, generator=torch.Generator().manual_seed(13)) * 10
    mask = torch.ones(24, dtype=torch.bool)
    mask[:4] = False
    sample_coords = _rigidly_moved(given, seed=17).unsqueeze(0).repeat(2, 1, 1)

    out, sampler, head = _forward_with_stubs(
        score_coordinates=ScoringCoordinates(given, mask),
        sample_coords=sample_coords,
    )

    assert sampler.n_calls == 1, "diffusion must run to fill the missing atoms"
    scored = out["X_pred_rollout_L"]
    assert torch.equal(scored[:, mask], given[mask].unsqueeze(0).repeat(2, 1, 1))
    # The filled atoms came from the sample, superposed into the input's frame.
    assert (scored[:, ~mask] - given[~mask]).abs().max() < 1e-3
    for i, seen in enumerate(head.seen):
        assert torch.equal(seen, scored[i].unsqueeze(0))


def test_forward_returns_the_scored_coordinates_as_the_rollout_output():
    # `trainers/rf3.py` copies X_pred_rollout_L into network_output["X_L"], which the
    # output writer and the clash metric then consume; pin that it is the scored tensor.
    given = torch.randn(24, 3, generator=torch.Generator().manual_seed(14))
    out, _, _ = _forward_with_stubs(
        score_coordinates=ScoringCoordinates(given, torch.ones(24, dtype=torch.bool))
    )
    assert out["X_pred_rollout_L"].shape == (2, 24, 3)
    assert out["early_stopped"] is False
    assert out["X_L"] is None  # unchanged contract: the trainer does the remap


def test_scoring_coordinates_reports_completeness():
    coords = torch.randn(5, 3)
    assert ScoringCoordinates(coords, torch.ones(5, dtype=torch.bool)).is_complete
    mask = torch.ones(5, dtype=torch.bool)
    mask[2] = False
    assert not ScoringCoordinates(coords, mask).is_complete


def test_scoring_coordinates_is_not_traversed_by_precision_conversion():
    # Lightning's `_FabricModule.forward` runs `precision.convert_input` over its
    # arguments, which casts every floating tensor to bfloat16 under the inference
    # autocast. That would displace every given atom by ~0.1 A, so the holder must be
    # opaque to `apply_to_collection`.
    from lightning.fabric.plugins.precision.utils import _convert_fp_tensor
    from lightning_utilities import apply_to_collection

    holder = ScoringCoordinates(torch.randn(4, 3), torch.ones(4, dtype=torch.bool))
    args = ((holder,), {"bare": torch.randn(4, 3)})
    converted = apply_to_collection(
        args, function=_convert_fp_tensor, dtype=torch.Tensor, dst_type=torch.bfloat16
    )

    assert converted[1]["bare"].dtype == torch.bfloat16, "sanity: bare tensors convert"
    assert converted[0][0] is holder
    assert converted[0][0].coords.dtype == torch.float32


def test_forward_rejects_scoring_together_with_early_stopping():
    from rf3.model.RF3 import RF3WithConfidence

    model = object.__new__(RF3WithConfidence)
    model.training = False
    with pytest.raises(AssertionError, match="Early stopping"):
        RF3WithConfidence.forward(
            model,
            input={"f": {}, "t": torch.zeros(1)},
            n_cycle=1,
            should_early_stop_fn=lambda **kwargs: (False, {}),
            score_coordinates=ScoringCoordinates(
                torch.randn(4, 3), torch.ones(4, dtype=torch.bool)
            ),
        )


def test_forward_rejects_scoring_in_training_mode():
    from rf3.model.RF3 import RF3WithConfidence

    model = object.__new__(RF3WithConfidence)
    model.training = True
    with pytest.raises(AssertionError, match="not supported during training"):
        RF3WithConfidence.forward(
            model,
            input={"f": {}, "t": torch.zeros(1)},
            n_cycle=1,
            score_coordinates=ScoringCoordinates(
                torch.randn(4, 3), torch.ones(4, dtype=torch.bool)
            ),
        )


def test_deque_contract_for_the_stubbed_recycling_generator():
    # Guards the stub above: `forward` drains the recycling generator with a maxlen-1
    # deque, so a one-element generator must still yield a usable final output.
    gen = iter([{"S_I": 1}])
    assert deque(gen, maxlen=1).pop() == {"S_I": 1}
