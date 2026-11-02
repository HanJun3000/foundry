import torch


def norm_seq_grad(grad: torch.Tensor) -> torch.Tensor:
    """
    Normalize the given gradient tensor similar to the JAX version.

    Steps:
      1. Compute the row-wise sum of squares.
      2. Count the number of rows with a nonzero sum (eff_L).
      3. Compute the global Frobenius norm of the entire tensor.
      4. Scale the gradient by sqrt(eff_L) / (global norm + epsilon).

    Args:
        grad (torch.Tensor): The gradient tensor with shape [L, D].

    Returns:
        torch.Tensor: The normalized gradient tensor of the same shape.
    """
    # Step 1: row-wise squared sum
    rowwise_sqsum = grad.pow(2).sum(dim=-1)

    # Step 2: Create a mask for nonzero rows and count them
    mask = rowwise_sqsum > 0
    eff_L = mask.sum().float()  # effective number of rows with nonzero gradient

    # Step 3: Compute the global Frobenius norm of the gradient
    gn = grad.norm(p=2)

    # Step 4: Compute the scaling factor
    scale = torch.sqrt(eff_L) / (gn + 1e-7)

    # Step 5: Return the scaled gradient
    return grad * scale


# Test the function with a simple example:
if __name__ == "__main__":
    # Construct a test gradient tensor of shape [3, 3] for simplicity
    # Let's assume we have 3 rows, and each row is 3-dimensional.
    grad = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # Row 0: squared sum = 1 + 4 + 9 = 14
            [0.0, 0.0, 0.0],  # Row 1: squared sum = 0 (zero row)
            [4.0, 5.0, 6.0],  # Row 2: squared sum = 16 + 25 + 36 = 77
        ]
    )

    # Expected:
    # Nonzero rows: row 0 and row 2, so eff_L = 2, sqrt(2) ~ 1.4142.
    # Global norm: sqrt(14 + 0 + 77) = sqrt(91) ~ 9.5394.
    # Scaling factor = 1.4142 / 9.5394 ~ 0.1482.
    # So, the normalized gradient should be grad * 0.1482 approximately.

    norm_grad = norm_seq_grad(grad)

    print("Original grad:")
    print(grad)
    print("\nNormalized grad:")
    print(norm_grad)

    # For verification, print the scaling factor computed
    rowwise_sqsum = grad.pow(2).sum(dim=-1)
    mask = rowwise_sqsum > 0
    eff_L = mask.sum().float()
    gn = grad.norm(p=2)
    scale = torch.sqrt(eff_L) / (gn + 1e-7)
    print("\nComputed scale factor:", scale.item())

    # Expected normalized grad approximately:
    # Row 0: [1*0.1482, 2*0.1482, 3*0.1482] ~ [0.1482, 0.2964, 0.4446]
    # Row 1: [0, 0, 0]
    # Row 2: [4*0.1482, 5*0.1482, 6*0.1482] ~ [0.5928, 0.7410, 0.8892]
    # So, the normalized grad should be:
    # [[0.1482, 0.2965, 0.4447],
    #  [0.0, 0.0, 0.0],
    #  [0.5930, 0.7412, 0.8895]]
