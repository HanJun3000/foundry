import torch
import torch.nn.functional as F


def ensure_probability(
    tensor, dim=-1, method="softmax", temperature=0.1, mask=None, force_apply=False
):
    """
    Ensures the tensor (L * 20) is a valid probability distribution.

    If not satisfied:
    - If method="softmax", applies softmax normalization.
    - If method="sum", removes zero elements and normalizes by sum.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to normalize.
        method (str): "softmax" or "sum" (default: "softmax").
        mask (torch.Tensor): Mask (1D) to apply before normalization.
        temperature (float): Temperature for softmax normalization.

    Returns:
        torch.Tensor: Valid probability distribution.
    """
    tensor = tensor.clone()  # Ensure tensor remains unmodified

    # Make a copy of mask to prevent unexpected changes
    mask_copy = mask.clone() if mask is not None else None

    # Ensure mask is properly broadcasted to match the normalization dimension
    if mask_copy is not None:
        if mask_copy.dim() != tensor.dim():
            mask_copy = mask_copy.unsqueeze(-1).expand_as(tensor)  # Broadcast mask

    # Step 1: Check if the tensor already satisfies probability conditions
    is_non_negative = (tensor >= 0).all()
    is_within_one = (tensor <= 1).all()
    sum_tensor = tensor.sum(dim=dim, keepdim=True)
    is_sum_one = torch.allclose(sum_tensor, torch.ones_like(sum_tensor), atol=1e-3)

    if is_non_negative and is_within_one and is_sum_one and not force_apply:
        return tensor  # Already valid

    # Step 2: Apply normalization
    if method == "softmax":
        norm_tensor = F.softmax(tensor / temperature, dim=dim)

    elif method == "sum":
        tensor[tensor < 0] = 0  # Remove negative values
        sum_tensor = tensor.sum(dim=dim, keepdim=True)
        valid_mask = sum_tensor > 0  # Prevent division by zero
        norm_tensor = tensor / sum_tensor.where(
            valid_mask, torch.tensor(1.0, device=tensor.device)
        )

    else:
        raise ValueError("Invalid method. Choose either 'softmax' or 'sum'.")

    # Apply mask only to selected sequences
    if mask_copy is not None:
        mask_copy = mask_copy.to(dtype=tensor.dtype)  # Convert to float
        return mask_copy * norm_tensor + (1 - mask_copy) * tensor

    return norm_tensor


if __name__ == "__main__":
    L = 2
    restype_tensor = torch.zeros(L, 20)
    restype_tensor[torch.arange(L), torch.zeros(L, dtype=torch.long)] = 1
    mask = torch.zeros(L, dtype=torch.bool)
    mask[0] = 1
    for i in range(10):
        restype_tensor = ensure_probability(
            restype_tensor,
            method="softmax",
            dim=-1,
            mask=mask,
            temperature=0.3,
            force_apply=True,
        )
        print(restype_tensor)
