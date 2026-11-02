from collections import defaultdict
from typing import Any

import biotite.structure as struc
import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.ml.encoding_definitions import (
    AF3SequenceEncoding,
)
from atomworks.ml.transforms.atom_array import (
    chain_instance_iter,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.template import blank_af3_template_features
from atomworks.ml.utils.geometry import apply_inverse_rigid, rigid_from_3_points
from atomworks.ml.utils.numpy import select_data_by_id
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from torch.nn.functional import normalize


def featurize_templates_like_af3_custom(
    atom_array: AtomArray,
    templates_by_chain: dict[str, list[dict[str, Any]]],
    sequence_encoding: AF3SequenceEncoding,
    gap_token: str = "<G>",
    allowed_chain_type: list[ChainType] = [ChainType.POLYPEPTIDE_L, ChainType.RNA],
    distogram_bins: torch.Tensor = torch.linspace(3.25, 50.75, 38),  # in Angstrom
) -> dict[str, torch.Tensor]:
    # Get the maximum number of templates for any chain, which will be the number of templates to fill
    n_templates = (
        max(
            len(templates_by_chain.get(chain_id, [])) for chain_id in templates_by_chain
        )
        if templates_by_chain
        else 0
    )
    n_templates = max(
        n_templates, 1
    )  # Ensure at least one template is filled (use a blank template if no templates)

    # Get full atom array token starts (useful for going from atom-level > token-level annotations)
    _a_token_starts = get_token_starts(atom_array)  # [n_token] (int)

    # Initialize features to fill
    _n_token = len(_a_token_starts)

    blank_af3_template = blank_af3_template_features(
        n_templates, _n_token, sequence_encoding.token_to_idx[gap_token]
    )
    res_type = blank_af3_template["template_restype"]  # [n_templates, n_token] (int)
    template_pseudo_beta_mask = blank_af3_template[
        "template_pseudo_beta_mask"
    ]  # [n_templates, n_token] (bool)
    template_backbone_frame_mask = blank_af3_template[
        "template_backbone_frame_mask"
    ]  # [n_templates, n_token] (bool)
    template_distogram = blank_af3_template[
        "template_distogram"
    ]  # [n_templates, n_token, n_token] (float)
    template_unit_vector = blank_af3_template[
        "template_unit_vector"
    ]  # [n_templates, n_token, n_token, 3] (float)

    # Fill the template features chain by chain and template by template ...
    for chain in chain_instance_iter(atom_array):
        # Check for allowable chain types
        if chain.chain_type[0] not in allowed_chain_type:
            # Only fill templates for proteins
            print(
                f"Chain {chain.chain_id[0]} type {chain.chain_type[0]} not in allowed_chain_type, skipping."
            )
            continue

        # Check for chains where templates exist
        chain_id = chain.chain_id[0]
        if chain_id not in templates_by_chain:
            # Early exit if there are no templates for this chain
            print(f"No templates for chain {chain_id}, skipping.")
            continue

        # Get chain token starts (useful for going from atom-level > token-level annotations)
        _c_token_starts = get_token_starts(chain)  # [n_token_in_chain] (int)
        # ... atomized tokens cannot be matched to templates
        if "atomize" in chain.get_annotation_categories():
            is_token_atomized = chain.atomize[
                _c_token_starts
            ]  # [n_token_in_chain] (bool)
        else:
            is_token_atomized = np.zeros_like(_c_token_starts, dtype=bool)
        matchable_query_chain_tokens = _c_token_starts[
            ~is_token_atomized
        ]  # [n_matchable_token_in_chain] (int)

        # Featurize the templates and insert into the template features
        # print(f'Processing chain {chain_id} with {len(templates_by_chain[chain_id])} templates.')
        for tmpl_idx, tmpl_data in enumerate(templates_by_chain[chain_id]):
            template = tmpl_data["atom_array"]

            # ========== FIX: Correct misalignment in aligned_query_res_idx ==========
            # When using input file templates, aligned_query_res_idx uses absolute res_id values.
            # However, matching requires within-chain relative indices: within_chain_res_idx.
            # Therefore, aligned_query_res_idx needs to be converted from absolute res_id to relative index.

            # Build a mapping from res_id to within_chain_res_idx for the query chain.
            query_res_ids = np.unique(chain.res_id)
            res_id_to_within_chain_idx = {}
            for res_id in query_res_ids:
                mask = chain.res_id == res_id
                if np.any(mask):
                    within_chain_idx = chain.within_chain_res_idx[mask][0]
                    res_id_to_within_chain_idx[res_id] = within_chain_idx

            # Check whether the template's aligned_query_res_idx needs conversion.
            template_aligned_values = np.unique(
                template.aligned_query_res_idx[template.aligned_query_res_idx >= 0]
            )
            query_within_chain_values = chain.within_chain_res_idx[
                matchable_query_chain_tokens
            ]

            # If none of the aligned_query_res_idx values are in the within_chain_res_idx of the query,
            # conversion is needed (this typically occurs when using input file templates).
            if len(template_aligned_values) > 0 and not np.any(
                np.isin(template_aligned_values, query_within_chain_values)
            ):
                print(
                    f"Converting template aligned_query_res_idx from absolute res_id to relative within_chain_res_idx for chain {chain_id}"
                )
                # Perform the conversion of aligned_query_res_idx
                corrected_aligned_idx = np.array(
                    [
                        res_id_to_within_chain_idx.get(res_id, -1)
                        for res_id in template.aligned_query_res_idx
                    ]
                )
                template.set_annotation("aligned_query_res_idx", corrected_aligned_idx)

            # ================================================================

            # Filter the template to only include tokens that are aligned to the query chain and that are not atomized
            # ... we use -1 as a placeholder query_res_idx for template tokens without alignment
            has_aligned_res_annotation = template.aligned_query_res_idx >= 0
            # ... find all template tokens that are aligned to the query chain
            has_match_in_query_chain = np.isin(
                template.aligned_query_res_idx,
                chain.within_chain_res_idx[matchable_query_chain_tokens],
            )
            # ... check there is at least one template token that is aligned to the query chain
            if not np.any(has_match_in_query_chain & has_aligned_res_annotation):
                # skip templates that do not have any aligned residues in the query
                # (e.g. because query chain was cropped and crop does not overlap with template)
                print(
                    f"Template {tmpl_idx} for chain {chain_id} has no aligned residues in the query chain, skipping."
                )
                continue
            # ... subset the template to only the relevant tokens
            template = template[has_match_in_query_chain & has_aligned_res_annotation]

            # Get template token starts (useful for going from atom-level > token-level annotations)
            _t_token_starts = get_token_starts(template)

            # Annotate the global `token_id` for the template tokens which will be used to match
            #  the template tokens to the query chain to fill the template features
            template_token_id = select_data_by_id(
                select_ids=template.aligned_query_res_idx[_t_token_starts],
                data_ids=chain.within_chain_res_idx[matchable_query_chain_tokens],
                data=chain.token_id[matchable_query_chain_tokens],
                axis=0,
            )  # [n_token_in_template] (int)
            # ... match based on global token ids
            _is_matched_token = np.isin(
                atom_array.token_id[_a_token_starts], template_token_id
            )  # [n_token] (bool)
            token_ids_to_fill = atom_array.token_id[_a_token_starts][
                _is_matched_token
            ]  # [n_matchable_token_in_template] (int)
            token_idxs_to_fill = np.where(_is_matched_token)[
                0
            ]  # [n_matchable_token_in_template] (int)

            # ... fill the res_type
            res_type[tmpl_idx, token_idxs_to_fill] = torch.as_tensor(
                sequence_encoding.encode(struc.get_residues(template)[1])
            )

            # ...fill the template_pseudo_beta_mask
            #   get information on whether the (pseudo) CB is resolved
            _is_cb = template.atom_name == "CB"
            _is_glycine_ca = (template.atom_name == "CA") & (template.res_name == "GLY")
            _is_pseudo_cb_resolved = (_is_cb | _is_glycine_ca) & (
                template.occupancy > 0
            )
            # ... spread it accross the token axis
            _has_pseudo_cb = struc.apply_residue_wise(
                template, data=_is_pseudo_cb_resolved, function=np.any
            )

            if np.any(_has_pseudo_cb):
                print(f"Template {tmpl_idx} for chain {chain_id} is valid")

            # ... fill the template_backbone_frame_mask
            _is_n_ca_c_resolved = (
                (template.atom_name == "CA")
                | (template.atom_name == "N")
                | (template.atom_name == "C") & (template.occupancy > 0)
            )
            _has_n_ca_c_resolved = (
                struc.apply_residue_wise(
                    template, data=(_is_n_ca_c_resolved), function=np.sum
                )
                == 3
            )
            template_backbone_frame_mask[tmpl_idx, token_idxs_to_fill] = (
                torch.as_tensor(_has_n_ca_c_resolved)
            )

            # ... fill the template_distogram
            template_coords = torch.tensor(template.coord)
            ix1, ix2 = np.ix_(
                token_ids_to_fill[_has_pseudo_cb], token_ids_to_fill[_has_pseudo_cb]
            )
            template_distogram[tmpl_idx, ix1.astype(int), ix2.astype(int)] = (
                torch.cdist(
                    template_coords[_is_pseudo_cb_resolved],
                    template_coords[_is_pseudo_cb_resolved],
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
            )

            # ... fill the template_unit_vector

            residues_with_resolved_n_ca_c = struc.spread_residue_wise(
                template, _has_n_ca_c_resolved
            )
            template_frames = rigid_from_3_points(
                x1=template_coords[
                    (template.atom_name == "N") & (residues_with_resolved_n_ca_c)
                ],
                x2=template_coords[
                    (template.atom_name == "CA") & (residues_with_resolved_n_ca_c)
                ],
                x3=template_coords[
                    (template.atom_name == "C") & (residues_with_resolved_n_ca_c)
                ],
            )  # (n_template_res, 3, 3), (n_template_res, 3)
            # ... get CA coords in the respective frames
            ca_coords_in_frames = apply_inverse_rigid(
                rigid=(
                    template_frames[0][:, None, :, :],
                    template_frames[1][:, None, :],
                ),
                points=template_coords[
                    (template.atom_name == "CA") & (residues_with_resolved_n_ca_c)
                ],
            )  # (n_template_res, n_template_res, 3)
            ca_direction_in_frames = normalize(ca_coords_in_frames, dim=-1, eps=1e-3)
            # ... reset diagonal to 0 (can be non-zero due to normalization & numerical error)
            ca_direction_in_frames[0, 0] = 0.0

            ix1, ix2 = np.ix_(
                token_ids_to_fill[_has_n_ca_c_resolved],
                token_ids_to_fill[_has_n_ca_c_resolved],
            )
            template_unit_vector[tmpl_idx, ix1.astype(int), ix2.astype(int)] = (
                ca_direction_in_frames
            )

    # ... bucketize the distogram
    template_distogram = torch.bucketize(
        template_distogram,
        boundaries=torch.as_tensor(
            distogram_bins,
            dtype=template_distogram.dtype,
            device=template_distogram.device,
        ),
    )
    n_bins = len(distogram_bins) + 1
    template_distogram = torch.nn.functional.one_hot(
        template_distogram, num_classes=n_bins
    ).to(torch.float32)  # We don't need int64 precision

    return {
        "template_restype": res_type,
        "template_pseudo_beta_mask": template_pseudo_beta_mask,
        "template_backbone_frame_mask": template_backbone_frame_mask,
        "template_distogram": template_distogram,
        "template_unit_vector": template_unit_vector,
    }


def add_input_file_template(
    atom_array: AtomArray,
):
    template = defaultdict(list)
    for chain in chain_instance_iter(atom_array):
        # Check for allowable chain types
        if chain.chain_type[0] not in [ChainType.POLYPEPTIDE_L, ChainType.RNA]:
            # Only fill templates for proteins
            continue

        # Check for chains where templates exist
        if np.sum(chain.is_input_file_templated) == 0:
            # Early exit if there are no templates for this chain
            continue

        chain_id = chain.chain_id[0]
        template_chain = atom_array[atom_array.is_input_file_templated]
        # add extra template annotations to the template
        # aligned_query_res_idx, alignment_confidence
        template_chain.set_annotation("aligned_query_res_idx", template_chain.res_id)
        template_chain.set_annotation(
            "alignment_confidence", np.ones(len(template_chain), dtype=float)
        )
        template[chain_id].append(
            {
                "id": None,
                "pdb_id": None,
                "chain_id": None,
                "template_lookup_id": None,
                "seq_similarity": 100.0,
                "atom_array": template_chain,
                "n_res": len(np.unique(template_chain.res_id)),
            }
        )
    return template


class AddInputFileTemplate(Transform):
    """
    If atoms from the input file have been marked as templates, add them to the template dictionary.
    This is useful for when users want to use a part of their design as a template using
    the template_selection_syntax argument in the inference script.
    """

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        template = add_input_file_template(atom_array)
        # Add the templates to the data
        if "template" in data:
            raise ValueError(
                "Template already exists in data. Cannot add input file template."
            )
        data["template"] = template
        return data
