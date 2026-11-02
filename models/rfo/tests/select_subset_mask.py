import torch


def select_subset_mask(mask: torch.Tensor, num_select: int) -> torch.Tensor:
    """
    Given a boolean mask of shape [N], randomly select a subset of indices where the mask is True,
    with size num_select. Return a new boolean mask that is True exactly at the selected indices,
    and False elsewhere.
    """
    # Get indices where mask is True.
    true_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if true_indices.numel() < num_select:
        raise ValueError(
            "Not enough True positions in the input mask to select the desired subset."
        )

    # Randomly permute and select num_select indices.
    perm = torch.randperm(len(true_indices))
    selected_indices = true_indices[perm][:num_select]

    # Create a new mask with False everywhere, then set True at the selected indices.
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[selected_indices] = True
    return new_mask


if __name__ == "__main__":
    # Example: original mask of length 8.
    orig_mask = torch.tensor([True, False, True, True, False, True, False, True])
    print("Original mask:", orig_mask.tolist())

    num_select = 2
    new_mask = select_subset_mask(orig_mask, num_select)
    print("New mask (selected subset):", new_mask.tolist())

    # Verify that new_mask is a subset of orig_mask:
    subset = (new_mask & ~orig_mask).sum().item()
    print("Number of True in new_mask not in orig_mask:", subset)

    # Also check that the number of True positions equals num_select.
    count_true = new_mask.sum().item()
    print("Number of True in new_mask:", count_true)
