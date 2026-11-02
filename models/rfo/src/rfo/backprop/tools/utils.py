import json
import re

import torch
import torch.nn.functional as F


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def index2mask(index, length):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[index] = True
    return mask


def ensure_probability(
    tensor, dim=-1, method="softmax", temperature=0.1, mask=None, force_apply=False
):
    """
    Ensures the tensor (L * 20) is a valid probability distribution.

    If not satisfied:
    - If method="softmax", applies softmax normalization.
    - If method="sum", removes zero elements and normalizes by sum.

    Args:
        tensor: Input tensor.
        dim: Dimension along which to normalize.
        method: "softmax" or "sum" (default: "softmax").
        mask: Mask (1D) to apply before normalization.
        temperature: Temperature for softmax normalization.
        force_apply: If True, apply normalization even if already valid.

    Returns:
        Valid probability distribution.
    """
    tensor = tensor.clone()

    mask = mask.clone() if mask is not None else None

    if mask is not None:
        if mask.dim() != tensor.dim():
            mask = mask.unsqueeze(-1).expand_as(tensor)

    is_non_negative = (tensor >= 0).all()
    is_within_one = (tensor <= 1).all()
    sum_tensor = tensor.sum(dim=dim, keepdim=True)
    is_sum_one = torch.allclose(sum_tensor, torch.ones_like(sum_tensor), atol=1e-3)

    if is_non_negative and is_within_one and is_sum_one and not force_apply:
        return tensor

    if method == "softmax":
        norm_tensor = F.softmax(tensor / temperature, dim=dim)

    elif method == "sum":
        tensor[tensor < 0] = 0
        sum_tensor = tensor.sum(dim=dim, keepdim=True)
        if sum_tensor.sum() <= 0:
            print(
                "Warning: Sum of prob is zero. Falling back to softmax normalization. (in ensure_probability function)"
            )
            return F.softmax(tensor / temperature, dim=dim)

        valid_mask = sum_tensor > 0
        norm_tensor = tensor / sum_tensor.where(
            valid_mask, torch.tensor(1.0, device=tensor.device)
        )

    else:
        raise ValueError("Invalid method. Choose either 'softmax' or 'sum'.")

    if mask is not None:
        mask = mask.to(dtype=tensor.dtype)
        return mask * norm_tensor + (1 - mask) * tensor

    return norm_tensor


def norm_seq_grad(grad: torch.Tensor) -> torch.Tensor:
    """
    Normalize the given gradient tensor.

    Args:
        grad: The gradient tensor of shape [L, D].

    Returns:
        The normalized gradient tensor of the same shape.
    """
    rowwise_sqsum = grad.pow(2).sum(dim=-1)

    mask = rowwise_sqsum > 0
    eff_L = mask.sum().float()

    gn = grad.norm(p=2)

    scale = torch.sqrt(eff_L) / (gn + 1e-7)

    return grad * scale


def select_subset_mask(mask: torch.Tensor, num_select: int) -> torch.Tensor:
    """
    Given a boolean mask of shape [N], randomly select a subset of indices where the mask is True,
    with size ``num_select``. Return a new boolean mask that is True exactly at the selected indices,
    and False elsewhere.
    """
    true_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if true_indices.numel() < num_select:
        raise ValueError(
            "Not enough True positions in the input mask to select the desired subset."
        )

    perm = torch.randperm(len(true_indices))
    selected_indices = true_indices[perm][:num_select]

    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[selected_indices] = True
    return new_mask


def natural_sort_key(s):
    """
    Splits the string into numeric and non-numeric parts, converting numeric parts
    to integers to ensure that '10' sorts after '2'.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", s)
    ]


def modelhub2af3(modelhub_inputs, output_path=None):
    """Convert modelhub inputs to AF3 format JSON."""
    af3_json = []
    for index, modelhub_input in enumerate(modelhub_inputs):
        entities = modelhub_input["components"]
        name = modelhub_input["name"]
        af3_entities = {
            "name": f"{name}",
            "sequences": [],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }
        for entity in entities:
            item = {
                "protein": {
                    "id": entity["chain_id"],
                    "sequence": entity["seq"],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                    "unpairedMsaPath": entity.get("msa_path", ""),
                }
            }
            af3_entities["sequences"].append(item)
        af3_json.append(af3_entities)
    if output_path:
        write_json(af3_json, output_path)
    else:
        return af3_json


def compute_saliency_pos_aa(
    gradient, seq_mask=None, threshold=0.04, min_num_candidates=20
):
    """
    Only keep positions with negative gradient. If the number of candidates
    is less than ``min_num_candidates``, gradually relax the threshold until
    enough candidates are found or threshold reaches 0.
    """
    grad_neg = gradient.clone()
    grad_neg[grad_neg >= 0] = 0.0
    mag_neg = grad_neg.abs()

    if seq_mask is not None:
        mag_pool = mag_neg[seq_mask]
    else:
        mag_pool = mag_neg.view(-1)

    keep_mask = mag_neg >= threshold
    if seq_mask is not None:
        keep_mask &= seq_mask.unsqueeze(-1)

    if keep_mask.sum() < min_num_candidates:
        sorted_vals = torch.sort(mag_pool.flatten(), descending=True).values
        if len(sorted_vals) >= min_num_candidates:
            threshold = sorted_vals[min_num_candidates - 1].item()
            keep_mask = mag_neg >= threshold
            if seq_mask is not None:
                keep_mask &= seq_mask.unsqueeze(-1)

    return keep_mask, grad_neg


def compute_saliency(gradient):
    """Given token gradients (L, 20) -> saliency (L,)."""
    grad_norm = torch.linalg.vector_norm(gradient, dim=-1)
    grad_norm = torch.relu(grad_norm)
    return grad_norm / grad_norm.sum()


def format_loss(loss, **kwargs):
    """Return a dictionary with the loss and additional information."""
    return_dict = {}
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return_dict["loss"] = float(loss)
    for key, value in kwargs.items():
        return_dict[key] = value
    return return_dict
