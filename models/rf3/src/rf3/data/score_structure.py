"""Structure scoring mode: run the confidence head on user-supplied coordinates.

Scoring mode runs RF3's normal trunk (sequence, MSA, templates) but evaluates the
confidence head on the coordinates of the *input structure* instead of on the
model's own diffusion samples. The output is pLDDT / PAE / PDE / pTM / ipTM "as if
RF3 had predicted this structure".

Why this works for RF3
----------------------
The trunk never sees coordinates; only :class:`~rf3.model.layers.af3_auxiliary_heads.ConfidenceHead`
does, via its ``X_pred_L`` argument. In the released checkpoint
(``use_Cb_distances: false``) that argument is consumed in exactly one place: a
pairwise distance matrix over the per-token *representative* atoms, binned into 40
bins from 3.25 to 50.75 Angstrom. pLDDT and ``exp_resolved`` are projected from the
single representation alone, and PAE/PDE from the pair representation after the
confidence pairformer.

Two consequences shape this module:

- Scoring is exactly invariant under a global rotation or translation of the input,
  because a distance matrix is. The input needs no alignment to anything.
- Only the **representative atoms** must carry real coordinates for the score to
  mean anything. The representative atom is CB for amino acids (CA for glycine and
  for unknown residues), C4 for purines, C2 for pyrimidines, and every atom of an
  atomized/ligand token. A missing side-chain atom cannot change the score; a
  missing CB invalidates that token.

Coordinates and masks
---------------------
``ground_truth["coord_atom_lvl"]`` already holds the input file's coordinates in
the model's atom order (see :class:`atomworks.ml.transforms.feature_aggregation.af3.AggregateFeaturesLikeAF3`),
index-for-index with ``X_L`` and with ``pipeline_output["atom_array"]``.

It is **not** safe to use unmasked. Atoms absent from the input file are given NaN
coordinates by the atomworks CCD-template fill, and
:func:`rf3.utils.inference.apply_conformer_and_template_selections` then replaces
every NaN with ``-1``. Reading those entries would place a dense fake cluster near
the origin and corrupt the distance matrix. Hence :func:`record_input_presence`
captures a boolean "this atom came from the file" annotation *before* the
substitution, and :func:`build_score_coordinates` zeroes everything outside it.

References:
    - AlphaFold 3 Supplementary Information, section 4.3 (confidence head) and
      Algorithm 28 (weighted rigid alignment).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray
from jaxtyping import Bool, Float

from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

#: Boolean atom-array annotation marking atoms whose coordinates come from the input
#: file (as opposed to atoms added from a CCD template to complete a residue).
PRESENCE_ANNOTATION = "is_present_in_input"

#: Minimum number of given atoms needed to define a rigid superposition.
MIN_ATOMS_FOR_ALIGNMENT = 3

#: How many missing representative atoms to name in the summary and the warning.
MAX_REPORTED_MISSING_ATOMS = 50


@dataclass
class ScoringSpec:
    """Opt-in scoring configuration for a single example.

    Args:
        enabled: Whether to score the input structure instead of predicting one.
        min_representative_atom_coverage: Refuse to score when the fraction of token
            representative atoms the input provides falls below this value. Those are
            the only coordinates the confidence head reads, so the rest of the score
            reflects a diffusion sample rather than the given structure. Experimental
            structures routinely miss a few (disordered loops, N-terminal residues), so
            the default of ``0.0`` never refuses and only warns; raise it to make a
            campaign reject poorly covered inputs.
        structure_path: Reserved: path to a separate structure file supplying the
            coordinates, for scoring a sequence-based (JSON) input. Not yet
            implemented.
        chain_ids: Reserved: one file chain per model chain, in model chain order.
            Not yet implemented.
    """

    enabled: bool = False
    min_representative_atom_coverage: float = 0.0
    structure_path: Path | None = None
    chain_ids: list[str] | None = None

    _JSON_KEYS = frozenset(
        {
            "enabled",
            "min_representative_atom_coverage",
            "structure_path",
            "structurePath",
            "chain_ids",
            "chainIds",
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_representative_atom_coverage <= 1.0:
            raise ValueError(
                "`min_representative_atom_coverage` must be a fraction in [0, 1], got "
                f"{self.min_representative_atom_coverage}."
            )

    @classmethod
    def from_json_value(cls, value: Any, *, example_id: str = "") -> "ScoringSpec":
        """Build a spec from the ``score_structure`` field of a JSON job.

        Accepts ``true`` / ``false`` (score the input file as-is), or a mapping with
        any of ``enabled``, ``min_representative_atom_coverage``, ``structure_path`` /
        ``structurePath``, ``chain_ids`` / ``chainIds``.

        Args:
            value: The raw JSON value.
            example_id: Example name, used only in error messages.

        Returns:
            The parsed spec.

        Raises:
            ValueError: On an unsupported type or an unknown key.
            NotImplementedError: For the not-yet-supported ``structure_path`` /
                ``chain_ids`` form.
        """
        where = f" for example {example_id!r}" if example_id else ""

        if isinstance(value, bool):
            return cls(enabled=value)

        if not isinstance(value, Mapping):
            raise ValueError(
                f"`score_structure`{where} must be a boolean or a mapping, "
                f"got {type(value).__name__}."
            )

        unknown = set(value) - cls._JSON_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in `score_structure`{where}: {sorted(unknown)}. "
                f"Supported keys: {sorted(cls._JSON_KEYS)}."
            )

        structure_path = value.get("structure_path", value.get("structurePath"))
        chain_ids = value.get("chain_ids", value.get("chainIds"))
        if structure_path is not None or chain_ids is not None:
            raise NotImplementedError(
                f"`score_structure.structure_path` / `.chain_ids`{where} are not yet "
                "supported. Scoring currently takes the coordinates from the input "
                "CIF/PDB itself, so the file's sequence must be the intended "
                "sequence. To score coordinates from a separate file, pass that file "
                "as the input."
            )

        return cls(
            enabled=bool(value.get("enabled", True)),
            min_representative_atom_coverage=float(
                value.get("min_representative_atom_coverage", 0.0)
            ),
        )


def record_input_presence(atom_array: AtomArray) -> AtomArray:
    """Annotate which atoms carry coordinates from the input file.

    Must be called while absent atoms still have NaN coordinates, i.e. before any
    sentinel substitution. Adds the boolean :data:`PRESENCE_ANNOTATION` annotation
    in place and returns the same array for convenience.

    Args:
        atom_array: Atom array straight out of the parser.

    Returns:
        The same atom array, with :data:`PRESENCE_ANNOTATION` set.
    """
    atom_array.set_annotation(
        PRESENCE_ANNOTATION, ~np.isnan(atom_array.coord).any(axis=-1)
    )
    return atom_array


def build_score_coordinates(
    pipeline_output: dict,
) -> tuple[Float[torch.Tensor, "L 3"], Bool[torch.Tensor, " L"]]:
    """Extract the scoring coordinates and mask in the model's atom order.

    Coordinates come from ``ground_truth["coord_atom_lvl"]``. The mask comes from
    the :data:`PRESENCE_ANNOTATION` annotation on ``atom_array`` when present, and
    otherwise falls back to ``ground_truth["mask_atom_lvl"]`` (the occupancy mask),
    which is more conservative: it zeroes a whole residue when any of its N/CA/C
    frame atoms is unresolved.

    Coordinates outside the mask are forced to zero so that a ``-1`` sentinel can
    never be read by a downstream consumer that forgets the mask.

    Args:
        pipeline_output: The featurized example, i.e. the output of the RF3
            transform pipeline.

    Returns:
        A pair ``(coords [L, 3] float32, mask [L] bool)``.

    Raises:
        KeyError: If ``ground_truth`` coordinates are missing.
        ValueError: If the mask and coordinate lengths disagree, or no atom is given.
    """
    ground_truth = pipeline_output.get("ground_truth", {})
    if "coord_atom_lvl" not in ground_truth:
        raise KeyError(
            "`ground_truth.coord_atom_lvl` is missing from the featurized example; "
            "cannot score a structure without input coordinates."
        )

    coords = ground_truth["coord_atom_lvl"].detach().clone().float()

    atom_array = pipeline_output.get("atom_array")
    if atom_array is not None and PRESENCE_ANNOTATION in set(
        atom_array.get_annotation_categories()
    ):
        mask = torch.as_tensor(
            np.asarray(atom_array.get_annotation(PRESENCE_ANNOTATION), dtype=bool),
            device=coords.device,
        )
    else:
        ranked_logger.warning(
            f"`{PRESENCE_ANNOTATION}` annotation not found on the atom array; falling "
            "back to the occupancy mask (`ground_truth.mask_atom_lvl`), which also "
            "masks residues with unresolved backbone frame atoms."
        )
        mask = ground_truth["mask_atom_lvl"].detach().clone().bool()

    if mask.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Scoring mask length ({mask.shape[0]}) does not match the number of "
            f"model atoms ({coords.shape[0]})."
        )
    if not bool(mask.any()):
        raise ValueError(
            "No input coordinates to score: every model atom is absent from the "
            "input structure."
        )

    # Never let a `-1` sentinel (or a NaN) escape through an unmasked read.
    coords = torch.where(mask.unsqueeze(-1), coords, torch.zeros_like(coords))
    return torch.nan_to_num(coords), mask


def overlay_score_coordinates(
    X_sample_DLC: Float[torch.Tensor, "D L 3"],
    X_score_LC: Float[torch.Tensor, "L 3"],
    score_mask_L: Bool[torch.Tensor, " L"],
) -> Float[torch.Tensor, "D L 3"]:
    """Fill atoms missing from the input structure from the diffusion samples.

    For each sample: rigidly superpose the sample onto the given atoms (Kabsch, in
    float64), then overwrite the given atoms with the input coordinates. Atoms
    absent from the input keep their sampled positions, now expressed in the input
    structure's frame, so they sit at sensible bond lengths instead of at the origin.

    Args:
        X_sample_DLC: Diffusion samples, ``[D, L, 3]``.
        X_score_LC: Input coordinates in the model's atom order, ``[L, 3]``.
        score_mask_L: Which atoms the input provides, ``[L]``.

    Returns:
        The overlaid coordinates, ``[D, L, 3]``, in ``X_sample_DLC``'s dtype.

    Raises:
        ValueError: On a shape mismatch, or when fewer than
            :data:`MIN_ATOMS_FOR_ALIGNMENT` atoms are given.
    """
    if X_sample_DLC.ndim != 3 or X_sample_DLC.shape[-1] != 3:
        raise ValueError(
            f"Expected samples of shape [D, L, 3], got {tuple(X_sample_DLC.shape)}."
        )
    n_sample, n_atom, _ = X_sample_DLC.shape
    if tuple(X_score_LC.shape) != (n_atom, 3):
        raise ValueError(
            f"Expected scoring coordinates of shape [{n_atom}, 3], got "
            f"{tuple(X_score_LC.shape)}."
        )
    if tuple(score_mask_L.shape) != (n_atom,):
        raise ValueError(
            f"Expected a scoring mask of shape [{n_atom}], got "
            f"{tuple(score_mask_L.shape)}."
        )

    mask = score_mask_L.to(device=X_sample_DLC.device, dtype=torch.bool)
    n_given = int(mask.sum())
    if n_given < MIN_ATOMS_FOR_ALIGNMENT:
        raise ValueError(
            f"Need at least {MIN_ATOMS_FOR_ALIGNMENT} given atoms to superpose the "
            f"diffusion samples onto the input structure, got {n_given}."
        )

    out_dtype = X_sample_DLC.dtype
    target = (
        X_score_LC.to(device=X_sample_DLC.device, dtype=torch.float64)
        .unsqueeze(0)
        .repeat(n_sample, 1, 1)
    )
    # `weighted_rigid_align` moves its `X_gt_L` argument onto `X_L`, so the sample is
    # the moving body and the given atoms are the fixed one.
    aligned = weighted_rigid_align(
        X_L=target,
        X_gt_L=X_sample_DLC.to(torch.float64),
        X_exists_L=mask,
    )

    overlaid = aligned.clone()
    overlaid[:, mask] = target[:, mask]
    return overlaid.to(out_dtype)


def _residue_group_ids(atom_array: AtomArray) -> np.ndarray:
    """Return a per-atom integer id that is constant within a residue."""
    chain = np.asarray(
        atom_array.chain_iid
        if "chain_iid" in set(atom_array.get_annotation_categories())
        else atom_array.chain_id
    ).astype(str)
    keys = np.stack(
        [chain, np.asarray(atom_array.res_id).astype(str)],
        axis=-1,
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return np.asarray(inverse).reshape(-1)


def summarize_scoring(
    *,
    example_id: str,
    atom_array: AtomArray,
    score_mask_L: Bool[torch.Tensor, " L"],
    rep_atom_idxs: torch.Tensor,
    diffusion_skipped: bool,
    structure_path: PathLike | str | None = None,
) -> dict:
    """Build the payload written to ``<example_id>_scoring_info.json``.

    Args:
        example_id: Example name.
        atom_array: The featurized atom array (model atom order).
        score_mask_L: Which atoms the input provides, ``[L]``.
        rep_atom_idxs: Per-token representative atom indices, ``[I]``.
        diffusion_skipped: Whether the diffusion rollout was skipped entirely.
        structure_path: Path the coordinates came from, if known.

    Returns:
        A JSON-serializable summary: overall and per-chain atom coverage,
        representative-atom coverage, which residues are missing their representative
        atom (up to :data:`MAX_REPORTED_MISSING_ATOMS`), residue counts, and whether
        diffusion ran.
    """
    mask = score_mask_L.detach().cpu().numpy().astype(bool)
    rep_idxs = rep_atom_idxs.detach().cpu().numpy().astype(int)
    rep_mask = mask[rep_idxs]

    residue_ids = _residue_group_ids(atom_array)
    chain_labels = np.asarray(
        atom_array.chain_iid
        if "chain_iid" in set(atom_array.get_annotation_categories())
        else atom_array.chain_id
    ).astype(str)

    def _coverage(selected: np.ndarray) -> dict:
        n_total = int(selected.sum())
        if n_total == 0:
            return {"n_given": 0, "n_total": 0, "coverage": None}
        n_given = int(mask[selected].sum())
        return {
            "n_given": n_given,
            "n_total": n_total,
            "coverage": round(n_given / n_total, 6),
        }

    is_representative = np.zeros_like(mask)
    is_representative[rep_idxs] = True

    chains: dict[str, dict] = {}
    for label in dict.fromkeys(chain_labels.tolist()):
        in_chain = chain_labels == label
        chain_residues = residue_ids[in_chain]
        chain_atom_mask = mask[in_chain]
        unique_residues = np.unique(chain_residues)
        n_mapped = 0
        n_complete = 0
        for residue in unique_residues:
            in_residue = chain_residues == residue
            if chain_atom_mask[in_residue].any():
                n_mapped += 1
            if chain_atom_mask[in_residue].all():
                n_complete += 1

        chains[str(label)] = {
            "atoms": _coverage(in_chain),
            "representative_atoms": _coverage(in_chain & is_representative),
            "n_residues": int(unique_residues.size),
            "n_residues_mapped": n_mapped,
            "n_residues_complete": n_complete,
        }

    # Name the residues whose representative atom is absent: those are the positions
    # whose confidence reflects a diffusion sample rather than the given structure.
    missing_rep_idxs = rep_idxs[~rep_mask]
    missing_representative_atoms = [
        {
            "chain": str(chain_labels[i]),
            "res_id": int(atom_array.res_id[i]),
            "res_name": str(atom_array.res_name[i]),
            "atom_name": str(atom_array.atom_name[i]),
        }
        for i in missing_rep_idxs[:MAX_REPORTED_MISSING_ATOMS].tolist()
    ]

    return {
        "example_id": example_id,
        "structure_path": str(structure_path) if structure_path is not None else None,
        "diffusion_skipped": bool(diffusion_skipped),
        "atoms": {
            "n_given": int(mask.sum()),
            "n_total": int(mask.size),
            "coverage": round(float(mask.mean()), 6),
        },
        "representative_atoms": {
            "n_given": int(rep_mask.sum()),
            "n_total": int(rep_mask.size),
            "coverage": round(float(rep_mask.mean()), 6),
        },
        "missing_representative_atoms": missing_representative_atoms,
        "n_missing_representative_atoms": int(missing_rep_idxs.size),
        "chains": chains,
    }


def log_scoring_summary(info: dict) -> None:
    """Log the scoring summary, one line per chain plus a totals line."""
    path = info.get("structure_path") or "<input>"
    for label, chain in info["chains"].items():
        atoms = chain["atoms"]
        reps = chain["representative_atoms"]
        ranked_logger.info(
            f"score_structure {path} chain {label} -> model chain {label}: "
            f"{chain['n_residues_mapped']}/{chain['n_residues']} residues mapped "
            f"({chain['n_residues_complete']} complete), "
            f"{atoms['n_given']}/{atoms['n_total']} atoms, "
            f"{reps['n_given']}/{reps['n_total']} representative atoms."
        )
    atoms = info["atoms"]
    reps = info["representative_atoms"]
    ranked_logger.info(
        f"score_structure {path}: {atoms['n_given']}/{atoms['n_total']} atoms given "
        f"({atoms['coverage']:.1%}), {reps['n_given']}/{reps['n_total']} "
        f"representative atoms ({reps['coverage']:.1%}); "
        f"diffusion {'skipped' if info['diffusion_skipped'] else 'run to fill gaps'}."
    )


def check_scoring_coverage(
    info: dict, *, min_representative_atom_coverage: float = 0.0
) -> None:
    """Warn about, and optionally refuse, incomplete representative-atom coverage.

    Representative atoms are the only coordinates the confidence head reads, so a gap
    there means part of the reported score describes a diffusion sample rather than
    the given structure. Gaps elsewhere do not affect the score at all and are filled
    by :func:`overlay_score_coordinates`.

    Experimental structures routinely miss a few representative atoms — disordered
    loops and N-terminal residues often have no CB — so the default never refuses and
    only warns, naming the affected residues. Raise
    ``min_representative_atom_coverage`` to make a campaign reject poorly covered
    inputs instead.

    Args:
        info: A :func:`summarize_scoring` payload.
        min_representative_atom_coverage: Refuse when coverage falls below this
            fraction. Defaults to ``0.0`` (never refuse).

    Raises:
        ValueError: When representative-atom coverage is below the threshold.
    """
    reps = info["representative_atoms"]
    n_missing = info["n_missing_representative_atoms"]
    if n_missing == 0:
        return

    named = ", ".join(
        f"{atom['chain']}/{atom['res_name']}{atom['res_id']}/{atom['atom_name']}"
        for atom in info["missing_representative_atoms"][:10]
    )
    if n_missing > 10:
        named += f", ... (+{n_missing - 10} more)"
    message = (
        f"{n_missing}/{reps['n_total']} token representative atoms (CB for amino "
        "acids, CA for glycine, C4/C2 for nucleotides, every ligand atom) are absent "
        f"from the input structure for {info['example_id']!r}: {named}. These are the "
        "only coordinates the confidence head reads, so their positions are taken "
        "from a diffusion sample and the confidences at those tokens partly describe "
        "RF3's own guess rather than the given structure."
    )

    coverage = reps["coverage"]
    if coverage is not None and coverage < min_representative_atom_coverage:
        raise ValueError(
            f"{message} Representative-atom coverage {coverage:.1%} is below the "
            f"required {min_representative_atom_coverage:.1%}."
        )
    ranked_logger.warning(message)
